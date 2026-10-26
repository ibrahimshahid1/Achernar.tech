// Hardware abstraction modules
// Real camera/mount/focuser support via OpenLiveStacker C++ drivers

pub mod camera;
pub mod capabilities;
pub mod discovery;
pub mod dslr_controls;
// The OLS mount and focuser wrappers now run only behind the equipment
// layer's legacy adapters (`equipment/legacy`), and the old filter-wheel
// stub is superseded by the native wheel. They are deleted as their own work
// package once the native drivers pass acceptance (owner decision O2); until
// then their unused parts are expected.
#[allow(dead_code)]
pub mod filter_wheel;
#[allow(dead_code)]
pub mod focuser;
pub mod health;
#[allow(dead_code)]
pub mod mount;
pub mod ols_ffi;
pub mod raw_decode;
pub mod sensors;
