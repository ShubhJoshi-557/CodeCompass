import type { NextAuthOptions } from "next-auth";
import GitHubProvider from "next-auth/providers/github";
import GitHub from "next-auth/providers/github"

const clientId = process.env.GITHUB_ID;
const clientSecret = process.env.GITHUB_SECRET;

if (!clientId || !clientSecret) {
  throw Error("Missing clientId or clientSecret");
}

export const authOptions: NextAuthOptions = {
  providers: [
    GitHubProvider({
      clientId: clientId,
      clientSecret: clientSecret,
    }),
  ],
};