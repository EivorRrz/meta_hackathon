from __future__ import annotations

from support_env.models import Category, ExpectedSupportAction, Ticket

ALL_TICKETS: tuple[Ticket, ...] = (
    Ticket(
        id="t_payment_failed",
        message="My payment failed twice when upgrading to Pro.",
        history=(),
        expected_category=Category.BILLING,
        expected_action=ExpectedSupportAction.TROUBLESHOOT,
        expected_resolution=(
            "Issue recorded as billing-001: retry with another card or bank auth window, "
            "then confirm invoice settles within 24h."
        ),
        expected_response=(
            "Thanks for flagging the failed payment—I'll verify the billing profile and "
            "outline the quickest retry steps."
        ),
        optimal_steps=4,
    ),
    Ticket(
        id="t_duplicate_charge",
        message="I was charged twice for the same subscription this month.",
        history=(),
        expected_category=Category.BILLING,
        expected_action=ExpectedSupportAction.REFUND,
        expected_resolution=(
            "Duplicate charge approved for reversal: refund queued to original payment "
            "method within 5 business days."
        ),
        expected_response=(
            "I'm sorry for the duplicate charge—I'll confirm both charges and start the "
            "refund process now."
        ),
        optimal_steps=4,
    ),
    Ticket(
        id="t_login_locked",
        message="Login fails and says my account is locked.",
        history=(),
        expected_category=Category.TECHNICAL,
        expected_action=ExpectedSupportAction.TROUBLESHOOT,
        expected_resolution=(
            "Security unlock applied; password reset sent. Please set a new password and "
            "re-enable 2FA if prompted."
        ),
        expected_response=(
            "I can help unlock this safely—I'll validate the account signals and send a "
            "controlled reset."
        ),
        optimal_steps=4,
    ),
    Ticket(
        id="t_refund_request",
        message="Please refund my last invoice—I no longer use the product.",
        history=(),
        expected_category=Category.BILLING,
        expected_action=ExpectedSupportAction.REFUND,
        expected_resolution=(
            "Refund approved per policy for unused period; credit will post to the card "
            "on file within 5 business days."
        ),
        expected_response=(
            "I've reviewed your subscription timeline and will process the refund per "
            "policy."
        ),
        optimal_steps=4,
    ),
    Ticket(
        id="t_slow_app",
        message="The desktop app is extremely slow after the latest update.",
        history=(),
        expected_category=Category.TECHNICAL,
        expected_action=ExpectedSupportAction.TROUBLESHOOT,
        expected_resolution=(
            "Performance fix bundle 2.14.3 + cache reset steps issued; confirm latency "
            "after restart and telemetry upload."
        ),
        expected_response=(
            "Thanks for the detailed report—I'll route this to our performance checklist "
            "and send mitigation steps."
        ),
        optimal_steps=4,
    ),
    Ticket(
        id="t_vague_complaint",
        message="This whole service is unacceptable. Fix it.",
        history=(),
        expected_category=Category.GENERAL,
        expected_action=ExpectedSupportAction.ESCALATE,
        expected_resolution=(
            "Escalated to leadership queue gen-esc-991 with full transcript for policy "
            "review and follow-up."
        ),
        expected_response=(
            "I hear you—I'm escalating this with urgency so the right owner can respond "
            "with a concrete plan."
        ),
        optimal_steps=5,
    ),
    Ticket(
        id="t_angry_duplicate",
        message="This is ridiculous—I'm furious. You charged me twice and nobody helps.",
        history=(
            "Prior chat: agent suggested waiting 48h; charge still doubled on statement.",
        ),
        expected_category=Category.BILLING,
        expected_action=ExpectedSupportAction.REFUND,
        expected_resolution=(
            "Duplicate charge approved for reversal: refund queued to original payment "
            "method within 5 business days."
        ),
        expected_response=(
            "I'm sorry for the repeat charge and the delay—I'll confirm both charges now "
            "and expedite the refund."
        ),
        optimal_steps=5,
        history_ack_substrings=("sorry", "delay", "refund"),
    ),
    Ticket(
        id="t_already_tried_restart",
        message="App still crashes on startup. Already rebooted and reinstalled twice.",
        history=("User states: reboot + reinstall already completed this morning.",),
        expected_category=Category.TECHNICAL,
        expected_action=ExpectedSupportAction.TROUBLESHOOT,
        expected_resolution=(
            "Assigned crash bucket TS-4412: collect diagnostic bundle, disable third-"
            "party plugins, install hotfix build 2.14.3."
        ),
        expected_response=(
            "Thanks for doing the reboot and reinstall—that rules out the common startup "
            "path; next we'll pull diagnostics and apply the hotfix."
        ),
        optimal_steps=5,
        history_ack_substrings=("reboot", "reinstall"),
    ),
)


def ticket_by_id(ticket_id: str) -> Ticket:
    for t in ALL_TICKETS:
        if t.id == ticket_id:
            return t
    raise KeyError(ticket_id)
