// Hardware abstraction modules
// TODO: Implement INDI client, serial mount, USB camera interfaces

pub mod indi_client;
pub mod serial_mount;
pub mod usb_camera;

#[cfg(target_os = "linux")]
pub mod v4l_camera;
