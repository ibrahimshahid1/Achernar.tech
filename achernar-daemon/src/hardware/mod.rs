// Hardware abstraction modules
// Camera support via OpenLiveStacker C++ drivers (O1); every other device
// class lives in the native equipment layer (`equipment`)

pub mod camera;
pub mod capabilities;
pub mod discovery;
pub mod dslr_controls;
// The OLS Alpaca mount, the OLS focuser and the old filter-wheel stub were
// deleted at the end of Tier 1 once the native drivers passed acceptance
// (owner decision O2). `focuser` keeps only the older in-process simulator
// the tests use, and `mount` the OLS wrapper behind the interim INDI adapter
// (`equipment/legacy`, O3); their unused parts are expected.
#[allow(dead_code)]
pub mod focuser;
pub mod health;
#[allow(dead_code)]
pub mod mount;
pub mod ols_ffi;
pub mod raw_decode;
pub mod sensors;
