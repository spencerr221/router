//! Current-request output-length prediction and capacity reservation.

use crate::config::types::PredictionMode;

use super::prediction_profile::{OutputLengthKey, OutputPrediction};
use super::ProgramIdentity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputReservationSource {
    Explicit,
    Prediction,
}

impl OutputReservationSource {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Prediction => "prediction",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputReservation {
    pub tokens: Option<usize>,
    pub source: Option<OutputReservationSource>,
}

impl OutputReservation {
    pub(crate) fn with_factor(self, minimum_tokens: usize, factor: f64) -> Self {
        Self {
            tokens: self.tokens.map(|tokens| {
                let additional = tokens.saturating_sub(minimum_tokens);
                minimum_tokens.saturating_add((additional as f64 * factor).ceil() as usize)
            }),
            source: self.source,
        }
    }
}

pub(crate) fn resolve_reservation(
    mode: PredictionMode,
    explicit_tokens: Option<usize>,
    prediction: Option<&OutputPrediction>,
) -> OutputReservation {
    if let Some(tokens) = explicit_tokens {
        return OutputReservation {
            tokens: Some(tokens),
            source: Some(OutputReservationSource::Explicit),
        };
    }
    if mode == PredictionMode::Active {
        if let Some(prediction) = prediction.filter(|prediction| prediction.mature) {
            return OutputReservation {
                tokens: Some(prediction.tokens),
                source: Some(OutputReservationSource::Prediction),
            };
        }
    }
    OutputReservation {
        tokens: None,
        source: None,
    }
}

pub(crate) fn keys_for_request(identity: &ProgramIdentity) -> Vec<OutputLengthKey> {
    let profile_role = identity.spawn_reason().or_else(|| identity.agent_role());
    let exact = OutputLengthKey {
        workflow_profile_id: identity.workflow_profile_id().map(str::to_string),
        profile_version: identity.profile_version().map(str::to_string),
        model_pool: Some(identity.model_pool().to_string()),
        profile_role: profile_role.map(str::to_string),
    };
    let mut keys = vec![exact.clone()];

    let model_role = OutputLengthKey {
        workflow_profile_id: None,
        profile_version: None,
        model_pool: exact.model_pool.clone(),
        profile_role: exact.profile_role.clone(),
    };
    push_unique(&mut keys, model_role);

    push_unique(
        &mut keys,
        OutputLengthKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: exact.model_pool,
            profile_role: None,
        },
    );
    push_unique(
        &mut keys,
        OutputLengthKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: None,
            profile_role: None,
        },
    );
    keys
}

fn push_unique(keys: &mut Vec<OutputLengthKey>, key: OutputLengthKey) {
    if !keys.contains(&key) {
        keys.push(key);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PendingOutputPrediction {
    pub keys: Vec<OutputLengthKey>,
    pub prediction: Option<OutputPrediction>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::prediction_profile::PredictionSource;
    use serde_json::json;

    fn identity(agent_role: &str, spawn_reason: Option<&str>) -> ProgramIdentity {
        let request = json!({
            "vllm_xargs": {"agentic_context": {
                "program_id": "workflow:agent",
                "task_id": "workflow",
                "expected_resume": true,
                "agent_role": agent_role,
                "spawn_reason": spawn_reason,
                "workflow_profile_id": "autogen-swebench",
                "profile_version": "1"
            }}
        });
        ProgramIdentity::from_request(None, Some(&request), Some("qwen"))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn workflow_stage_is_used_as_profile_role_with_layered_backoff() {
        for role in ["planner", "explorer", "engineer", "reviewer", "verifier"] {
            let keys = keys_for_request(&identity("lead", Some(role)));

            assert_eq!(keys.len(), 4);
            assert_eq!(keys[0].profile_role.as_deref(), Some(role));
            assert_eq!(keys[1].workflow_profile_id, None);
            assert_eq!(keys[1].profile_role.as_deref(), Some(role));
            assert_eq!(keys[2].profile_role, None);
            assert_eq!(keys[3].model_pool, None);
        }
    }

    #[test]
    fn agent_role_is_the_fallback_when_workflow_stage_is_absent() {
        let keys = keys_for_request(&identity("subagent", None));

        assert_eq!(keys[0].profile_role.as_deref(), Some("subagent"));
    }

    fn prediction(tokens: usize, mature: bool) -> OutputPrediction {
        OutputPrediction {
            tokens,
            p50: tokens,
            p95: tokens,
            p99: tokens,
            support: 100,
            online_support: 0,
            backoff_level: 0,
            source: PredictionSource::Offline,
            mature,
        }
    }

    #[test]
    fn active_reservation_prefers_explicit_then_mature_prediction() {
        let predicted = prediction(512, true);
        assert_eq!(
            resolve_reservation(PredictionMode::Active, Some(256), Some(&predicted)),
            OutputReservation {
                tokens: Some(256),
                source: Some(OutputReservationSource::Explicit),
            }
        );
        assert_eq!(
            resolve_reservation(PredictionMode::Active, None, Some(&predicted)),
            OutputReservation {
                tokens: Some(512),
                source: Some(OutputReservationSource::Prediction),
            }
        );
        let short = prediction(64, true);
        assert_eq!(
            resolve_reservation(PredictionMode::Active, None, Some(&short)).tokens,
            Some(64)
        );
    }

    #[test]
    fn shadow_and_immature_active_predictions_use_the_default_fallback() {
        let mature = prediction(512, true);
        let immature = prediction(512, false);
        assert_eq!(
            resolve_reservation(PredictionMode::Shadow, None, Some(&mature)),
            OutputReservation {
                tokens: None,
                source: None,
            }
        );
        assert_eq!(
            resolve_reservation(PredictionMode::Active, None, Some(&immature)),
            OutputReservation {
                tokens: None,
                source: None,
            }
        );
    }

    #[test]
    fn reservation_factor_preserves_the_decode_buffer_floor() {
        let reservation = OutputReservation {
            tokens: Some(6_000),
            source: Some(OutputReservationSource::Explicit),
        };
        assert_eq!(reservation.with_factor(100, 0.0).tokens, Some(100));
        assert_eq!(reservation.with_factor(100, 0.25).tokens, Some(1_575));
        assert_eq!(reservation.with_factor(100, 0.5).tokens, Some(3_050));
        assert_eq!(reservation.with_factor(100, 0.75).tokens, Some(4_525));
        assert_eq!(reservation.with_factor(100, 1.0).tokens, Some(6_000));
        assert_eq!(
            OutputReservation {
                tokens: Some(64),
                source: Some(OutputReservationSource::Explicit),
            }
            .with_factor(100, 0.5)
            .tokens,
            Some(100)
        );
    }
}
