import React, { useState } from 'react';
import { useTelescope } from '../context/TelescopeProvider';
import { CaptureParams } from '../api/client';

/**
 * CameraControl Component
 * 
 * Shared UI component for camera control
 * Works in both Desktop (via LocalClient) and Web (via CloudClient)
 */
export const CameraControl: React.FC = () => {
    const telescope = useTelescope();

    const [exposure, setExposure] = useState<number>(5.0);
    const [gain, setGain] = useState<number>(100);
    const [binning, setBinning] = useState<number>(1);
    const [capturing, setCapturing] = useState<boolean>(false);

    const handleCapture = async () => {
        setCapturing(true);
        try {
            const params: CaptureParams = {
                exposure,
                gain,
                binning,
            };

            const image = await telescope.capture(params);
            console.log('Captured image:', image.id);
            // TODO: Display image
        } catch (error) {
            console.error('Capture failed:', error);
        } finally {
            setCapturing(false);
        }
    };

    const handleAbort = async () => {
        try {
            await telescope.abortCapture();
        } catch (error) {
            console.error('Abort failed:', error);
        }
    };

    return (
        <div className="camera-control">
            <h2>Camera Control</h2>

            <div className="control-group">
                <label>
                    Exposure (s):
                    <input
                        type="number"
                        value={exposure}
                        onChange={(e) => setExposure(parseFloat(e.target.value))}
                        step="0.1"
                        min="0.001"
                    />
                </label>
            </div>

            <div className="control-group">
                <label>
                    Gain:
                    <input
                        type="number"
                        value={gain}
                        onChange={(e) => setGain(parseInt(e.target.value))}
                        min="0"
                        max="300"
                    />
                </label>
            </div>

            <div className="control-group">
                <label>
                    Binning:
                    <select value={binning} onChange={(e) => setBinning(parseInt(e.target.value))}>
                        <option value={1}>1x1</option>
                        <option value={2}>2x2</option>
                        <option value={4}>4x4</option>
                    </select>
                </label>
            </div>

            <div className="button-group">
                <button onClick={handleCapture} disabled={capturing}>
                    {capturing ? 'Capturing...' : 'Capture'}
                </button>
                <button onClick={handleAbort} disabled={!capturing}>
                    Abort
                </button>
            </div>
        </div>
    );
};
