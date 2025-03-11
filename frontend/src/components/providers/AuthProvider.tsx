"use client";

import { SessionProvider, SessionProviderProps } from "next-auth/react";

const AuthProvider = ({ session, children }: SessionProviderProps) => {
    return (
        <SessionProvider session={session}>
            {children}
        </SessionProvider>
    )
}

export default AuthProvider