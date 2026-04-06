//! Vision loop — Section 8.1.
//! Runs at 1–5 Hz, publishing SharedObservation to all subsystems.

use super::shared_observation::SharedObservation;
use tokio::sync::broadcast;

pub struct VisionLoop {
    publisher: broadcast::Sender<SharedObservation>,
    rate_hz: f64,
}

impl VisionLoop {
    pub fn new(rate_hz: f64) -> Self {
        let (tx, _) = broadcast::channel(64);
        Self { publisher: tx, rate_hz }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SharedObservation> {
        self.publisher.subscribe()
    }

    /// Main vision loop — processes frames and publishes observations.
    pub async fn run_vision_loop(&self) -> crate::error::Result<()> {
        todo!("Implement vision loop at {} Hz — Section 8.1", self.rate_hz)
    }
}
