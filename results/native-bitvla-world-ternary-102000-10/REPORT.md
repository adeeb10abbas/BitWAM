# Ternary-head BitWAM at 2,000 joint updates

The full-budget primary BitWAM checkpoint completed all 10 ordered LIBERO-10 smoke
rollouts, one trial per task. Correct-target cosine reached 0.9445 and the
correct-versus-shuffled action-conditioning gap reached 0.2146, compared with roughly
`5e-4` before contrastive joint training. Action L1 remained 0.0313.

This checkpoint meets the preregistered retention gate in the engineering screen and
shows that a ternary predictive head can shape a native ternary VLA without destroying
closed-loop control. It does not establish a success-rate improvement over action-only,
which also scored 10/10. The paired 50-rollout evaluation is the next evidence tier.
