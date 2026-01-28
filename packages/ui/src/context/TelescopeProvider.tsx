import React, { createContext, useContext, ReactNode } from 'react';
import { TelescopeClient } from '../api/client';

/**
 * Telescope Context
 * Provides dependency injection for TelescopeClient across the component tree
 */
const TelescopeContext = createContext<TelescopeClient | null>(null);

export interface TelescopeProviderProps {
    client: TelescopeClient;
    children: ReactNode;
}

/**
 * TelescopeProvider
 * 
 * Wraps the application with the appropriate client implementation:
 * - Desktop app uses LocalClient
 * - Web app uses CloudClient
 * 
 * @example
 * // Desktop App
 * <TelescopeProvider client={new LocalClient(port)}>
 *   <App />
 * </TelescopeProvider>
 * 
 * @example
 * // Web App
 * <TelescopeProvider client={new CloudClient()}>
 *   <App />
 * </TelescopeProvider>
 */
export const TelescopeProvider: React.FC<TelescopeProviderProps> = ({
    client,
    children
}) => {
    return (
        <TelescopeContext.Provider value={client}>
            {children}
        </TelescopeContext.Provider>
    );
};

/**
 * useTelescope Hook
 * 
 * Provides access to the TelescopeClient instance
 * Must be used within a TelescopeProvider
 * 
 * @throws Error if used outside TelescopeProvider
 * 
 * @example
 * function CameraControl() {
 *   const telescope = useTelescope();
 *   
 *   const handleCapture = async () => {
 *     const image = await telescope.capture({ exposure: 5.0 });
 *   };
 *   
 *   return <button onClick={handleCapture}>Capture</button>;
 * }
 */
export const useTelescope = (): TelescopeClient => {
    const context = useContext(TelescopeContext);

    if (!context) {
        throw new Error(
            'useTelescope must be used within a TelescopeProvider. ' +
            'Wrap your component tree with <TelescopeProvider client={client}>'
        );
    }

    return context;
};
