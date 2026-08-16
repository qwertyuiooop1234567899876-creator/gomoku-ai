from __future__ import annotations

import unittest

from engine.proof_search import ProofState
from engine.search_diagnostics import compose_search_reason
from engine.search_types import RootSafetyProbeResult


class TestSearchReasonComposition(unittest.TestCase):
    def test_dynamic_review_and_unknown_proof_are_reported(self) -> None:
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=100,
            main_rank_stable=True,
            completed_depth=5,
            nodes=20,
            candidates=(),
            selection_basis="frontier_balance",
        )

        reason = compose_search_reason(
            "PVS 搜索完成",
            expansion_reason="unverified_advantage",
            expansion_hold_applied=False,
            root_vcf_scan=None,
            mate_scores_quarantined=True,
            defense_risk_override=False,
            root_safety_probe=probe,
            root_safety_applied=True,
            final_proof_checked=True,
            final_proof_state=ProofState.UNKNOWN.value,
            final_proof_completed=False,
            final_proof_rejected=(),
        )

        self.assertIn("全盘生存候选扩展", reason)
        self.assertIn("Mate 分已降级", reason)
        self.assertIn("攻防前沿净增益改选", reason)
        self.assertTrue(reason.endswith("安全性未确认"))

    def test_proof_rejection_and_survival_are_reported_in_order(self) -> None:
        reason = compose_search_reason(
            "防守搜索完成",
            expansion_reason=None,
            expansion_hold_applied=False,
            root_vcf_scan=None,
            mate_scores_quarantined=False,
            defense_risk_override=True,
            root_safety_probe=None,
            root_safety_applied=False,
            final_proof_checked=True,
            final_proof_state=ProofState.PROVEN_LOSS.value,
            final_proof_completed=True,
            final_proof_rejected=((7, 7),),
        )

        self.assertIn("更低的对手威胁风险纠正", reason)
        self.assertIn("已淘汰可证败着", reason)
        self.assertTrue(reason.endswith("通过 Proof 生存确认"))


if __name__ == "__main__":
    unittest.main()
