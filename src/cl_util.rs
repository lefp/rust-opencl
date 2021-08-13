use ocl::{Platform, Device, DeviceType};

pub fn any_platform_with_substr(substr: &str) -> Option<Platform> {
    let substr = &substr.to_ascii_lowercase();
    Platform::list().into_iter()
        .find(|plat| {
            plat.name().unwrap_or_else(|_| {
                    println!("WARN: ignoring nameless platform");
                    "".to_string()
                })
                .to_ascii_lowercase()
                .contains(substr)
        })
}
pub fn any_device_with_substr(substr: &str, platform: &Platform) -> Option<Device> {
    let substr = &substr.to_ascii_lowercase();
    Device::list_all(platform).expect("failed to list devices").into_iter()
        .find(|dev| {
            dev.name().unwrap_or_else(|_| {
                println!("WARN: ignoring nameless device");
                "".to_string()
            })
            .to_ascii_lowercase()
            .contains(substr)
        })
}
pub fn any_gpu_device(platform: &Platform) -> Option<Device> {
    Device::list(platform, Some(DeviceType::new().gpu()))
        .expect("failed to list devices")
        .into_iter().next()
}