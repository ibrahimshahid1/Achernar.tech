//! Latest-frame selection for preview. Never replay an old connection's image.
use crate::hardware::camera::FrameData;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PreviewUpdate {
    Frame,
    Clear,
    Unchanged,
}

#[derive(Default)]
pub(crate) struct PreviewCursor {
    last: Option<(u64, i32)>,
    empty_sent: bool,
}

impl PreviewCursor {
    pub(crate) fn next(
        &mut self,
        frame: Option<&FrameData>,
        generation: u64,
        streaming: bool,
    ) -> PreviewUpdate {
        let frame = frame.filter(|frame| streaming && frame.stream_generation == generation);
        match frame {
            Some(frame) => {
                let identity = (frame.stream_generation, frame.frame_counter);
                if self.last == Some(identity) {
                    return PreviewUpdate::Unchanged;
                }
                self.last = Some(identity);
                self.empty_sent = false;
                PreviewUpdate::Frame
            }
            None if !self.empty_sent => {
                self.last = None;
                self.empty_sent = true;
                PreviewUpdate::Clear
            }
            None => PreviewUpdate::Unchanged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::ols_ffi::{OlsBayerType, OlsStreamType};

    fn frame(generation: u64, counter: i32) -> FrameData {
        FrameData {
            stream_generation: generation,
            frame_counter: counter,
            unix_timestamp: 1.0,
            width: 2,
            height: 2,
            format: OlsStreamType::Mono8,
            bayer: OlsBayerType::Na,
            data: vec![0; 4],
            source: None,
            saturation_adu: None,
            sensor_descriptor: None,
        }
    }

    #[test]
    fn sends_each_source_frame_once_and_clears_even_if_old_pixels_remain() {
        let mut cursor = PreviewCursor::default();
        let old = frame(1, 10);
        assert_eq!(cursor.next(Some(&old), 1, true), PreviewUpdate::Frame);
        assert_eq!(cursor.next(Some(&old), 1, true), PreviewUpdate::Unchanged);
        assert_eq!(cursor.next(Some(&old), 1, false), PreviewUpdate::Clear);
        assert_eq!(cursor.next(Some(&old), 1, false), PreviewUpdate::Unchanged);
    }

    #[test]
    fn rejects_retired_generation_and_accepts_reconnect_counter_reset() {
        let mut cursor = PreviewCursor::default();
        let old = frame(1, 0);
        assert_eq!(cursor.next(Some(&old), 1, true), PreviewUpdate::Frame);
        assert_eq!(cursor.next(Some(&old), 2, true), PreviewUpdate::Clear);
        assert_eq!(
            cursor.next(Some(&frame(2, 0)), 2, true),
            PreviewUpdate::Frame
        );
        assert_eq!(cursor.next(None, 2, true), PreviewUpdate::Clear);
    }
}
