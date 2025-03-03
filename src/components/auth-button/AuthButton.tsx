"use client";

import { signIn, signOut, useSession } from "next-auth/react";
import { Button } from "../ui/button";

const AuthButton = () => {
  const { data: session } = useSession();
  console.log(session);
  return (
    <div>
      {session?.user?.email ? (
        <Button variant="ghost" onClick={() => signOut()}>
          Sign out
        </Button>
      ) : (
        <Button variant="ghost" onClick={() => signIn()}>
          Sign in
        </Button>
      )}

    </div>
  );
};

export default AuthButton;
