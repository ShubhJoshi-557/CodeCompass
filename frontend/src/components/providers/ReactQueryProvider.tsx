"use client"; // ✅ This must be a Client Component

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactNode, useState } from "react";

export default function ReactQueryProvider({ children }: { children: ReactNode }) {
  const [queryClient] = useState(() => new QueryClient()); // Prevents re-creation on every render

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
