import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Edit } from "lucide-react";
import Link from "next/link";
import { Button } from "../ui/button";
import { UserTab } from "./UserTab";

export function AppSidebar() {
  return (
    <Sidebar className="border-none">
      <Link href="/">
        <SidebarHeader className="m-auto p-5 text-2xl">
          {"CodeCompass </>"}
        </SidebarHeader>
      </Link>

      <SidebarContent>
        <SidebarMenu>
          <SidebarMenuItem className="px-3 pt-2">
            <Link href="/">
              <Button className="font-semibold">
                <Edit />
                New Chat
              </Button>
            </Link>
          </SidebarMenuItem>
        </SidebarMenu>

        <SidebarGroup>
          <SidebarGroupLabel>Repository</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {/* {["projects"].map((project) => (
                <SidebarMenuItem key={project.name}>
                  <SidebarMenuButton asChild>
                    <a href={project.url}>
                      <project.icon />
                      <span>{project.name}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))} */}
              <SidebarMenuItem>
                <SidebarMenuButton>Codebase</SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Recent Searches</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Link href="/q/dbqkiyhgkyg">DB Query</Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Link href="/q/apiqkiyhgkyg">API Request Query</Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Link href="/q/docqkiyhgkyg">Documentation Query</Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <UserTab />
    </Sidebar>
  );
}
