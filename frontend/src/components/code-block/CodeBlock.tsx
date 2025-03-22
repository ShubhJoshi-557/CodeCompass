"use client";

import React from "react";

import { CodeBlock } from "@/components/ui/code-block";

export function CodeBlockDemo() {
  const code = `def add(a, b):
    """
    Adds two numbers and returns the result.

    Parameters:
    a (int or float): The first number.
    b (int or float): The second number.

    Returns:
    int or float: The sum of a and b.

    Raises:
    TypeError: If either a or b is not an int or float.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both a and b must be int or float.")
    
    return a + b`;

  return (
    <div className="max-w-3xl mx-auto w-full">
      <CodeBlock
        language="python"
        filename=" "
        // highlightLines={[9, 13, 14, 18]}
        code={code}
      />
    </div>
  );
}
