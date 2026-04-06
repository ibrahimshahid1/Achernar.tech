import React from "react";

/** Session summary card. */
export function SessionSummary({ sessionId }: { sessionId: string }) {
    return <div id={`session-${sessionId}`}>Session Summary — stub</div>;
}

/** List of frames in a session. */
export function FrameList({ sessionId }: { sessionId: string }) {
    return <div id={`frame-list-${sessionId}`}>Frame List — stub</div>;
}
