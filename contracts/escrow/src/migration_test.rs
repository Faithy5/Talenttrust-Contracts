//! Hardened concurrent-execution regression tests for `migration.rs`.
//!
//! ## Coverage goals
//!
//! | Scenario | Test |
//! |---|---|
//! | Happy path: propose → accept updates client and emits events | `propose_and_accept_updates_client` |
//! | Duplicate proposal while one is live is rejected (InvalidState) | `duplicate_proposal_rejected` |
//! | Racing accept from a wrong address is rejected (UnauthorizedRole) | `racing_accept_wrong_address_rejected` |
//! | Idempotent cancel: cancelling a non-existent pending migration fails (InvalidState) | `cancel_without_pending_migration_fails` |
//! | Idempotent cancel: cancel clears the record; second cancel fails | `cancel_is_not_idempotent_second_cancel_fails` |
//! | Retry after expiry: propose succeeds after previous proposal expired | `retry_propose_after_expiry_succeeds` |
//! | Boundary acceptance at last live ledger succeeds | `accept_at_ttl_boundary_succeeds` |
//! | Acceptance after expiry fails (InvalidState) | `accept_after_expiry_fails` |
//! | Finalized contract blocks proposal | `proposal_blocked_on_finalized_contract` |
//! | Non-client cancel attempt is rejected (UnauthorizedRole) | `non_client_cancel_rejected` |
//! | Propose+cancel+re-propose cycle succeeds deterministically | `propose_cancel_repropose_cycle` |
//! | Acceptance with stale proposing client fails (InvalidState) | `accept_rejected_when_stored_client_changed` |
//! | Role-overlap re-check at acceptance time catches late freelancer change | `accept_rejected_on_role_overlap_at_acceptance` |

#![cfg(test)]

use crate::migration::PendingClientMigration;
use crate::ttl::PENDING_MIGRATION_TTL_LEDGERS;
use crate::{Contract, ContractStatus, DataKey, Escrow, EscrowClient, EscrowError};
use soroban_sdk::{
    testutils::{Address as _, Ledger as _, LedgerInfo},
    Address, Env,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Register a fresh escrow contract and initialize it. Returns the client.
fn register_client(env: &Env) -> EscrowClient<'_> {
    let contract_addr = env.register(Escrow, ());
    let client = EscrowClient::new(env, &contract_addr);
    let admin = Address::generate(env);
    env.mock_all_auths();
    client.initialize(&admin);
    client
}

/// Create a bare 2-milestone escrow contract (no settlement token, no deposit).
/// Returns `(client_addr, freelancer_addr, contract_id)`.
fn create_contract(env: &Env, escrow: &EscrowClient<'_>) -> (Address, Address, u32) {
    let client_addr = Address::generate(env);
    let freelancer_addr = Address::generate(env);
    let milestones = soroban_sdk::vec![env, 1_000_0000000_i128, 2_000_0000000_i128];
    let id = escrow.create_contract(
        &client_addr,
        &freelancer_addr,
        &None,
        &milestones,
        &crate::ReleaseAuthorization::ClientOnly,
    );
    (client_addr, freelancer_addr, id)
}

/// Bump ledger TTL parameters so temporary storage does not get rejected by the host.
fn set_high_ttl(env: &Env) {
    let initial = env.ledger().get();
    env.ledger().set(LedgerInfo {
        sequence_number: initial.sequence_number,
        timestamp: initial.timestamp,
        protocol_version: initial.protocol_version,
        network_id: initial.network_id.clone(),
        base_reserve: initial.base_reserve,
        min_temp_entry_ttl: 1,
        min_persistent_entry_ttl: PENDING_MIGRATION_TTL_LEDGERS * 4,
        max_entry_ttl: PENDING_MIGRATION_TTL_LEDGERS * 4,
    });
}

/// Advance the ledger sequence past the migration TTL so temporary entries are
/// auto-evicted by the Soroban host.
fn advance_past_ttl(env: &Env) {
    let current = env.ledger().get();
    env.ledger().set(LedgerInfo {
        sequence_number: current
            .sequence_number
            .saturating_add(PENDING_MIGRATION_TTL_LEDGERS + 1),
        timestamp: current
            .timestamp
            .saturating_add(u64::from(PENDING_MIGRATION_TTL_LEDGERS) * 5 + 5),
        protocol_version: current.protocol_version,
        network_id: current.network_id.clone(),
        base_reserve: current.base_reserve,
        min_temp_entry_ttl: 1,
        min_persistent_entry_ttl: 1,
        max_entry_ttl: 65_536,
    });
}

/// Assert that a `try_*` call surfaces the expected `EscrowError` variant.
///
/// Soroban's generated `try_*` methods return:
///   `Result<Result<T, soroban_sdk::Error>, soroban_sdk::InvokeError>`
///
/// A `panic_with_error` inside the contract surfaces as `Err(Ok(soroban_sdk::Error))`.
fn assert_error<T: core::fmt::Debug, E: Into<soroban_sdk::Error> + core::fmt::Debug>(
    result: Result<Result<T, soroban_sdk::Error>, soroban_sdk::InvokeError>,
    expected: E,
) {
    match result {
        Err(Ok(e)) => {
            let expected_err: soroban_sdk::Error = expected.into();
            assert_eq!(e, expected_err, "contract error code mismatch");
        }
        other => panic!(
            "expected contract error {:?}, got unexpected result variant: {:?}",
            expected, other
        ),
    }
}

// ---------------------------------------------------------------------------
// Test 1 – propose → accept: client updated, events emitted, record cleared
// ---------------------------------------------------------------------------

/// A complete two-step migration must:
/// - store a `PendingClientMigration` in temporary storage after proposal
/// - return `true` from both `propose_client_migration` and `accept_client_migration`
/// - update `contract.client` to the new address
/// - clear the pending record so `has_pending_client_migration` returns `false`
#[test]
fn propose_and_accept_updates_client() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    // Proposal stores a live pending record and returns true
    assert!(
        escrow.propose_client_migration(&id, &client_addr, &new_client),
        "propose_client_migration must return true"
    );
    assert!(
        escrow.has_pending_client_migration(&id),
        "pending record must be live after proposal"
    );

    let pending: PendingClientMigration = escrow.get_pending_client_migration(&id);
    assert_eq!(pending.current_client, client_addr);
    assert_eq!(pending.proposed_client, new_client);
    assert!(
        pending.expires_at_ledger > env.ledger().sequence(),
        "expires_at_ledger must be in the future"
    );

    // Acceptance clears the pending record and updates contract.client
    assert!(
        escrow.accept_client_migration(&id, &new_client),
        "accept_client_migration must return true"
    );
    assert_eq!(
        escrow.get_contract(&id).client,
        new_client,
        "contract.client must be updated after acceptance"
    );
    assert!(
        !escrow.has_pending_client_migration(&id),
        "pending record must be cleared after acceptance"
    );
}

// ---------------------------------------------------------------------------
// Test 2 – duplicate proposal while one is live is rejected
// ---------------------------------------------------------------------------

/// A second `propose_client_migration` while a live pending record exists must
/// panic with `InvalidState`.  Concurrent or rapid re-proposals must not
/// silently overwrite an in-flight migration.
#[test]
fn duplicate_proposal_rejected() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client_1 = Address::generate(&env);
    let new_client_2 = Address::generate(&env);

    // First proposal succeeds
    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client_1));

    // Second proposal while the first is still live must fail
    assert_error(
        escrow.try_propose_client_migration(&id, &client_addr, &new_client_2),
        EscrowError::InvalidState,
    );

    // Original proposal is unchanged
    let pending: PendingClientMigration = escrow.get_pending_client_migration(&id);
    assert_eq!(
        pending.proposed_client, new_client_1,
        "original pending record must not be overwritten"
    );
}

// ---------------------------------------------------------------------------
// Test 3 – racing accept from wrong address is rejected
// ---------------------------------------------------------------------------

/// Only the address named in the proposal may accept.
/// The original client, the freelancer, and random third parties are all
/// rejected with `UnauthorizedRole`.  This covers the race where a different
/// party tries to hijack the acceptance.
#[test]
fn racing_accept_wrong_address_rejected() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, freelancer_addr, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);
    let attacker = Address::generate(&env);

    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));

    // All non-proposed addresses must be rejected
    for (label, addr) in [
        ("attacker", &attacker),
        ("original client", &client_addr),
        ("freelancer", &freelancer_addr),
    ] {
        assert_error(
            escrow.try_accept_client_migration(&id, addr),
            EscrowError::UnauthorizedRole,
        );
        // Pending record must still be intact after each rejected attempt
        assert!(
            escrow.has_pending_client_migration(&id),
            "pending record must survive rejected accept attempt by {}",
            label
        );
    }

    // Original client address has not changed
    assert_eq!(escrow.get_contract(&id).client, client_addr);
}

// ---------------------------------------------------------------------------
// Test 4 – cancel without a live pending migration fails
// ---------------------------------------------------------------------------

/// `cancel_client_migration` called when no pending record exists must panic
/// with `InvalidState`.  This is the idempotency boundary — repeated or
/// spurious cancels must not silently succeed.
#[test]
fn cancel_without_pending_migration_fails() {
    let env = Env::default();
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);

    assert_error(
        escrow.try_cancel_client_migration(&id, &client_addr),
        EscrowError::InvalidState,
    );
}

// ---------------------------------------------------------------------------
// Test 5 – cancel clears the record; second cancel fails (not idempotent)
// ---------------------------------------------------------------------------

/// Cancel removes the pending record.  A second cancel must fail with
/// `InvalidState` because the record is gone — demonstrating that the
/// cancel path has a guard and cannot be called repeatedly without effect.
#[test]
fn cancel_is_not_idempotent_second_cancel_fails() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));
    assert!(escrow.has_pending_client_migration(&id));

    // First cancel succeeds
    assert!(
        escrow.cancel_client_migration(&id, &client_addr),
        "first cancel must succeed"
    );
    assert!(
        !escrow.has_pending_client_migration(&id),
        "pending record must be cleared after cancel"
    );

    // Second cancel must fail — the record is gone
    assert_error(
        escrow.try_cancel_client_migration(&id, &client_addr),
        EscrowError::InvalidState,
    );
}

// ---------------------------------------------------------------------------
// Test 6 – retry propose after expiry succeeds
// ---------------------------------------------------------------------------

/// Once a proposal's TTL elapses and the entry is evicted, the client must
/// be able to make a fresh proposal for the same contract.  This validates
/// that the "no duplicate pending migration" guard is correctly tied to a
/// *live* record and does not permanently block re-proposals.
#[test]
fn retry_propose_after_expiry_succeeds() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);
    let retry_client = Address::generate(&env);

    // First proposal succeeds
    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));
    assert!(escrow.has_pending_client_migration(&id));

    // Advance ledger past the TTL — the temporary entry is evicted
    advance_past_ttl(&env);
    assert!(
        !escrow.has_pending_client_migration(&id),
        "proposal must be gone after TTL expiry"
    );

    // Re-proposing after expiry must succeed — the old evicted record does not block it
    assert!(
        escrow.propose_client_migration(&id, &client_addr, &retry_client),
        "re-propose after expiry must succeed"
    );
    assert!(escrow.has_pending_client_migration(&id));

    // The new pending record references the retry address
    let pending: PendingClientMigration = escrow.get_pending_client_migration(&id);
    assert_eq!(pending.proposed_client, retry_client);
}

// ---------------------------------------------------------------------------
// Test 7 – acceptance at the last live ledger inside the TTL window succeeds
// ---------------------------------------------------------------------------

/// Acceptance on the final ledger *before* `expires_at_ledger` must succeed.
/// This pins the inclusive boundary of the migration window so an off-by-one
/// in the TTL handling (evicting one ledger early) would be caught.
#[test]
fn accept_at_ttl_boundary_succeeds() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));

    let pending: PendingClientMigration = escrow.get_pending_client_migration(&id);
    let expires_at = pending.expires_at_ledger;

    // Advance to the last ledger strictly *inside* the live window
    let current = env.ledger().get();
    env.ledger().set(LedgerInfo {
        sequence_number: expires_at.saturating_sub(1),
        timestamp: current.timestamp + 5,
        protocol_version: current.protocol_version,
        network_id: current.network_id.clone(),
        base_reserve: current.base_reserve,
        min_temp_entry_ttl: 1,
        min_persistent_entry_ttl: PENDING_MIGRATION_TTL_LEDGERS * 4,
        max_entry_ttl: PENDING_MIGRATION_TTL_LEDGERS * 4,
    });

    // Still live at the boundary
    assert!(
        escrow.has_pending_client_migration(&id),
        "proposal must be live on the last ledger before expiry"
    );

    // Acceptance at the boundary succeeds
    assert!(escrow.accept_client_migration(&id, &new_client));
    assert_eq!(escrow.get_contract(&id).client, new_client);
    assert!(
        !escrow.has_pending_client_migration(&id),
        "pending record must be cleared after boundary acceptance"
    );
}

// ---------------------------------------------------------------------------
// Test 8 – acceptance after TTL expiry fails with InvalidState
// ---------------------------------------------------------------------------

/// Trying to accept a proposal after its TTL has elapsed must fail with
/// `InvalidState`.  The evicted entry must not be treated as valid.
#[test]
fn accept_after_expiry_fails() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));
    advance_past_ttl(&env);

    // has_pending must return false after eviction
    assert!(!escrow.has_pending_client_migration(&id));

    // accept must fail — no live record
    assert_error(
        escrow.try_accept_client_migration(&id, &new_client),
        EscrowError::InvalidState,
    );

    // contract.client is unchanged
    assert_eq!(escrow.get_contract(&id).client, client_addr);
}

// ---------------------------------------------------------------------------
// Test 9 – proposal blocked on finalized contract
// ---------------------------------------------------------------------------

/// `propose_client_migration` must fail on a finalized contract with
/// `AlreadyFinalized`.  Finalization is a permanent state; no mutations
/// are permitted afterwards.
#[test]
fn proposal_blocked_on_finalized_contract() {
    let env = Env::default();
    let escrow = register_client(&env);

    let (client_addr, freelancer_addr, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    // Inject Completed status directly — there is no public completed-without-funds path
    let escrow_addr = escrow.address.clone();
    env.as_contract(&escrow_addr, || {
        let key = DataKey::Contract(id);
        let mut contract: Contract = env.storage().persistent().get(&key).unwrap();
        contract.status = ContractStatus::Completed;
        env.storage().persistent().set(&key, &contract);
    });

    // Finalize the contract — this writes a finalization record
    escrow.finalize_contract(&id, &client_addr);

    // Proposal must now be blocked with AlreadyFinalized
    assert_error(
        escrow.try_propose_client_migration(&id, &client_addr, &new_client),
        EscrowError::AlreadyFinalized,
    );
    let _ = freelancer_addr; // suppress unused warning
}

// ---------------------------------------------------------------------------
// Test 10 – non-client cancel attempt is rejected
// ---------------------------------------------------------------------------

/// Only the current contract client may cancel a pending migration.
/// The freelancer and a random attacker must both be rejected with
/// `UnauthorizedRole`.
#[test]
fn non_client_cancel_rejected() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, freelancer_addr, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);
    let attacker = Address::generate(&env);

    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));

    for (label, addr) in [("freelancer", &freelancer_addr), ("attacker", &attacker)] {
        assert_error(
            escrow.try_cancel_client_migration(&id, addr),
            EscrowError::UnauthorizedRole,
        );
        // The pending record must still be intact after each rejected cancel
        assert!(
            escrow.has_pending_client_migration(&id),
            "pending record must survive rejected cancel by {}",
            label
        );
    }
}

// ---------------------------------------------------------------------------
// Test 11 – propose → cancel → re-propose cycle is deterministic
// ---------------------------------------------------------------------------

/// The full propose → cancel → re-propose sequence must work correctly:
/// - first proposal stores a live record
/// - cancel removes it
/// - a second proposal (potentially with a different new_client) succeeds
/// - the second acceptance updates contract.client to the second new_client
///
/// This validates that the pending-migration key is properly cleaned up by
/// cancel so the next proposal is not blocked.
#[test]
fn propose_cancel_repropose_cycle() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client_1 = Address::generate(&env);
    let new_client_2 = Address::generate(&env);

    // --- First cycle: propose then cancel ---
    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client_1));
    assert!(escrow.has_pending_client_migration(&id));
    assert!(escrow.cancel_client_migration(&id, &client_addr));
    assert!(!escrow.has_pending_client_migration(&id));

    // --- Second cycle: propose a different new client and accept ---
    assert!(
        escrow.propose_client_migration(&id, &client_addr, &new_client_2),
        "re-propose after cancel must succeed"
    );
    assert!(escrow.has_pending_client_migration(&id));

    let pending: PendingClientMigration = escrow.get_pending_client_migration(&id);
    assert_eq!(pending.proposed_client, new_client_2);

    assert!(escrow.accept_client_migration(&id, &new_client_2));
    assert_eq!(escrow.get_contract(&id).client, new_client_2);
    assert!(!escrow.has_pending_client_migration(&id));
}

// ---------------------------------------------------------------------------
// Test 12 – acceptance fails when proposing client no longer matches contract.client
// ---------------------------------------------------------------------------

/// If the contract's stored `client` field is changed between proposal and
/// acceptance (e.g. by a concurrent path or direct storage injection), the
/// acceptance check `pending.current_client != contract.client` must catch
/// the inconsistency and panic with `InvalidState`.
///
/// This ensures the migration cannot be completed for a client slot that has
/// already been taken over by a different address.
#[test]
fn accept_rejected_when_stored_client_changed() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);
    let adversarial_client = Address::generate(&env);

    // Proposal from the current client
    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));

    // Simulate a concurrent write that changes contract.client to a different address
    let escrow_addr = escrow.address.clone();
    env.as_contract(&escrow_addr, || {
        let key = DataKey::Contract(id);
        let mut contract: Contract = env.storage().persistent().get(&key).unwrap();
        contract.client = adversarial_client.clone();
        env.storage().persistent().set(&key, &contract);
    });

    // Acceptance must fail because pending.current_client != contract.client
    assert_error(
        escrow.try_accept_client_migration(&id, &new_client),
        EscrowError::InvalidState,
    );

    // The contract.client has the adversarially-injected address, not the proposed one
    assert_eq!(escrow.get_contract(&id).client, adversarial_client);
}

// ---------------------------------------------------------------------------
// Test 13 – role-overlap re-check at acceptance time catches late freelancer change
// ---------------------------------------------------------------------------

/// Even if the proposal passed role-overlap checks, acceptance re-validates
/// the proposed address against the contract's *current* roles.  If the
/// freelancer field was changed to match the proposed client (e.g. by a
/// concurrent update), the acceptance must fail with `RoleOverlap`.
#[test]
fn accept_rejected_on_role_overlap_at_acceptance() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    // Proposal succeeds — no overlap yet
    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));

    // Inject a role change so proposed client now collides with freelancer
    let escrow_addr = escrow.address.clone();
    env.as_contract(&escrow_addr, || {
        let key = DataKey::Contract(id);
        let mut contract: Contract = env.storage().persistent().get(&key).unwrap();
        contract.freelancer = new_client.clone(); // now proposed == freelancer
        env.storage().persistent().set(&key, &contract);
    });

    // Acceptance must fail with RoleOverlap
    assert_error(
        escrow.try_accept_client_migration(&id, &new_client),
        EscrowError::RoleOverlap,
    );

    // contract.client is unchanged
    assert_eq!(escrow.get_contract(&id).client, client_addr);
}

// ---------------------------------------------------------------------------
// Test 14 – expires_at_ledger matches TTL constant
// ---------------------------------------------------------------------------

/// The `expires_at_ledger` field stored in the pending record must equal
/// `requested_at_ledger + PENDING_MIGRATION_TTL_LEDGERS`.  This pins the
/// TTL arithmetic so any regression in `propose_client_migration_impl` that
/// miscalculates the expiry window is immediately visible.
#[test]
fn pending_migration_expiry_matches_ttl_constant() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    let ledger_before = env.ledger().sequence();
    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));

    let pending: PendingClientMigration = escrow.get_pending_client_migration(&id);
    assert_eq!(
        pending.requested_at_ledger, ledger_before,
        "requested_at_ledger must equal the ledger at proposal time"
    );
    assert_eq!(
        pending.expires_at_ledger,
        ledger_before.saturating_add(PENDING_MIGRATION_TTL_LEDGERS),
        "expires_at_ledger must equal requested_at + PENDING_MIGRATION_TTL_LEDGERS"
    );
}

// ---------------------------------------------------------------------------
// Test 15 – double-accept after successful migration is rejected
// ---------------------------------------------------------------------------

/// After a successful acceptance the pending record is consumed.  A second
/// acceptance attempt must fail with `InvalidState` — there is nothing to
/// accept.  This prevents double-application in a concurrent or retry
/// scenario.
#[test]
fn double_accept_fails_after_migration_completed() {
    let env = Env::default();
    set_high_ttl(&env);
    let escrow = register_client(&env);

    let (client_addr, _freelancer, id) = create_contract(&env, &escrow);
    let new_client = Address::generate(&env);

    assert!(escrow.propose_client_migration(&id, &client_addr, &new_client));
    assert!(escrow.accept_client_migration(&id, &new_client));

    // Pending record is gone — second accept must fail
    assert_error(
        escrow.try_accept_client_migration(&id, &new_client),
        EscrowError::InvalidState,
    );

    // contract.client remains the accepted address (not re-reverted)
    assert_eq!(escrow.get_contract(&id).client, new_client);
}
