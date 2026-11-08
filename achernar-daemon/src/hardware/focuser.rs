//! The older in-process focuser simulator.
//!
//! The OLS focuser wrapper that used to live here was deleted at the end of
//! Tier 1, with the OLS focuser itself, once the native Alpaca focuser passed
//! acceptance (owner decision O2). The `sim` focuser driver id runs the
//! equipment layer's kinematic simulator (`equipment/sim/focuser.rs`); this
//! one stays behind `equipment/legacy/sim_focuser.rs` for the tests that
//! were written against it.

#[derive(Debug)]
pub struct SimFocuser {
    position: i32,
    max_range: i32,
    moving: bool,
    /// Test hook: report moving until `abort`, like a stalled driver.
    #[cfg(test)]
    stuck_moving: bool,
}

impl SimFocuser {
    pub fn new(position: i32) -> Self {
        let max_range = 50000;
        Self {
            position: position.clamp(0, max_range),
            max_range,
            moving: false,
            #[cfg(test)]
            stuck_moving: false,
        }
    }

    /// Test hook: a device whose usable travel ends at `max_range`.
    #[cfg(test)]
    pub fn with_max_range(position: i32, max_range: i32) -> Self {
        let mut focuser = Self::new(position);
        focuser.max_range = max_range;
        focuser.position = position.clamp(0, max_range);
        focuser
    }

    /// Test hook: keep reporting `is_moving` until `abort` is called.
    #[cfg(test)]
    pub fn set_stuck_moving(&mut self, stuck: bool) {
        self.stuck_moving = stuck;
    }

    pub fn position(&self) -> crate::error::Result<i32> {
        Ok(self.position)
    }

    pub fn is_moving(&self) -> crate::error::Result<bool> {
        #[cfg(test)]
        if self.stuck_moving {
            return Ok(true);
        }
        Ok(self.moving)
    }

    pub fn is_absolute(&self) -> crate::error::Result<bool> {
        Ok(true)
    }

    pub fn max_range(&self) -> crate::error::Result<i32> {
        Ok(self.max_range)
    }

    pub fn move_absolute(&mut self, position: i32) -> crate::error::Result<()> {
        self.moving = true;
        self.position = position.clamp(0, self.max_range);
        self.moving = false;
        Ok(())
    }

    pub fn move_relative(&mut self, steps: i32) -> crate::error::Result<()> {
        self.move_absolute(self.position + steps)
    }

    pub fn abort(&mut self) -> crate::error::Result<()> {
        self.moving = false;
        #[cfg(test)]
        {
            self.stuck_moving = false;
        }
        Ok(())
    }
}
