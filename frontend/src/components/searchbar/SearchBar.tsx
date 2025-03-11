"use client";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { Textarea } from "@/components/ui/textarea";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { SearchButton } from "./SearchButton";

const FormSchema = z.object({
  query: z
    .string()
    .min(10, {
      message: "Query must be at least 10 characters.",
    })
    .max(200, {
      message: "Query must not be longer than 200 characters.",
    }),
});

export function SearchBar() {
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
  });

  function onSubmit(data: z.infer<typeof FormSchema>, searchType: string) {
    console.log(`Searching in: ${searchType}`, data);
    // Handle search logic based on searchType ("current-file" or "entire-codebase")
  }

  return (
    <Form {...form}>
      <form className="w-3xl space-y-6">
        <FormField
          control={form.control}
          name="query"
          render={({ field }) => (
            <FormItem>
              <div className="w-full max-h-[210px] mx-auto flex p-2 bg-white dark:bg-neutral-950">
                <div className="w-full p-2 border-none rounded-3xl bg-neutral-200 dark:bg-neutral-600">
                  <FormControl>
                    <Textarea
                      className="border-none outline-none resize-none max-h-[180px]"
                      placeholder="Search code, understand logic, and debug smarter..."
                      {...field}
                    ></Textarea>
                  </FormControl>
                </div>
                <SearchButton
                  onSearch={(searchType) =>
                    form.handleSubmit((data) => onSubmit(data, searchType))()
                  }
                />
              </div>
              <FormMessage className="mx-5" />
            </FormItem>
          )}
        />
      </form>
    </Form>
  );
}