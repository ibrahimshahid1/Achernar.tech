// API
export { LocalClient, CloudClient } from './api/client';
export type {
    TelescopeClient,
    CaptureParams,
    ImageData,
    StreamFrame,
    EquipmentStatus,
    FocuserAPI,
    MountAPI
} from './api/client';

// Context
export { TelescopeProvider, useTelescope } from './context/TelescopeProvider';
export type { TelescopeProviderProps } from './context/TelescopeProvider';

// Components
export { CameraControl } from './components/CameraControl';
