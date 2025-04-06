import LLMResponse from "@/components/llm-response/LLMResponse";
import { CodeBlock } from "@/components/ui/code-block";

const results = {
  results: [
    {
      filename: "git-mergetool.sh",
      filepath: "git-mergetool.sh",
      content:
        '#!/bin/sh\n#\n# This program resolves merge conflicts in git\n#\n# Copyright (c) 2006 Theodore Y. Ts\'o\n# Copyright (c) 2009-2016 David Aguilar\n#\n# This file is licensed under the GPL v2, or a later version\n# at the discretion of Junio C Hamano.\n#\n\nUSAGE=\'[--tool=tool] [--tool-help] [-y|--no-prompt|--prompt] [-g|--gui|--no-gui] [-O<orderfile>] [file to merge] ...\'\nSUBDIRECTORY_OK=Yes\nNONGIT_OK=Yes\nOPTIONS_SPEC=\nTOOL_MODE=merge\n. git-sh-setup\n. git-mergetool--lib\n\n# Returns true if the mode reflects a symlink\nis_symlink () {\n\ttest "$1" = 120000\n}\n\nis_submodule () {\n\ttest "$1" = 160000\n}\n\nlocal_present () {\n\ttest -n "$local_mode"\n}\n\nremote_present () {\n\ttest -n "$remote_mode"\n}\n\nbase_present () {\n\ttest -n "$base_mode"\n}\n\nmergetool_tmpdir_init () {\n\tif test "$(git config --bool mergetool.writeToTemp)" != true\n\tthen\n\t\tMERGETOOL_TMPDIR=.\n\t\treturn 0\n\tfi\n\tif MERGETOOL_TMPDIR=$(mktemp -d -t "git-mergetool-XXXXXX" 2>/dev/null)\n\tthen\n\t\treturn 0\n\tfi\n\tdie "error: mktemp is needed when \'mergetool.writeToTemp\' is true"\n}\n\ncleanup_temp_files () {\n\tif test "$1" = --save-backup\n\tthen\n\t\trm -rf -- "$MERGED.orig"\n\t\ttest -e "$BACKUP" && mv -- "$BACKUP" "$MERGED.orig"\n\t\trm -f -- "$LOCAL" "$REMOTE" "$BASE"\n\telse\n\t\trm -f -- "$LOCAL" "$REMOTE" "$BASE" "$BACKUP"\n\tfi\n\tif test "$MERGETOOL_TMPDIR" != "."\n\tthen\n\t\trmdir "$MERGETOOL_TMPDIR"\n\tfi\n}\n\ndescribe_file () {\n\tmode="$1"\n\tbranch="$2"\n\tfile="$3"\n\n\tprintf "  {%s}: " "$branch"\n\tif test -z "$mode"\n\tthen\n\t\techo "deleted"\n\telif is_symlink "$mode"\n\tthen\n\t\techo "a symbolic link -> \'$(cat "$file")\'"\n\telif is_submodule "$mode"\n\tthen\n\t\techo "submodule commit $file"\n\telif base_present\n\tthen\n\t\techo "modified file"\n\telse\n\t\techo "created file"\n\tfi\n}\n\nresolve_symlink_merge () {\n\twhile true\n\tdo\n\t\tprintf "Use (l)ocal or (r)emote, or (a)bort? "\n\t\tread ans || return 1\n\t\tcase "$ans" in\n\t\t[lL]*)\n\t\t\tgit checkout-index -f --stage=2 -- "$MERGED"\n\t\t\tgit add -- "$MERGED"\n\t\t\tcleanup_temp_files --save-backup\n\t\t\treturn 0\n\t\t\t;;\n\t\t[rR]*)\n\t\t\tgit checkout-index -f --stage=3 -- "$MERGED"\n\t\t\tgit add -- "$MERGED"\n\t\t\tcleanup_temp_files --save-backup\n\t\t\treturn 0\n\t\t\t;;\n\t\t[aA]*)\n\t\t\treturn 1\n\t\t\t;;\n\t\tesac\n\tdone\n}\n\nresolve_deleted_merge () {\n\twhile true\n\tdo\n\t\tif base_present\n\t\tthen\n\t\t\tprintf "Use (m)odified or (d)eleted file, or (a)bort? "\n\t\telse\n\t\t\tprintf "Use (c)reated or (d)eleted file, or (a)bort? "\n\t\tfi\n\t\tread ans || return 1\n\t\tcase "$ans" in\n\t\t[mMcC]*)\n\t\t\tgit add -- "$MERGED"\n\t\t\tif test "$merge_keep_backup" = "true"\n\t\t\tthen\n\t\t\t\tcleanup_temp_files --save-backup\n\t\t\telse\n\t\t\t\tcleanup_temp_files\n\t\t\tfi\n\t\t\treturn 0\n\t\t\t;;\n\t\t[dD]*)\n\t\t\tgit rm -- "$MERGED" > /dev/null\n\t\t\tcleanup_temp_files\n\t\t\treturn 0\n\t\t\t;;\n\t\t[aA]*)\n\t\t\tif test "$merge_keep_temporaries" = "false"\n\t\t\tthen\n\t\t\t\tcleanup_temp_files\n\t\t\tfi\n\t\t\treturn 1\n\t\t\t;;\n\t\tesac\n\tdone\n}\n\nresolve_submodule_merge () {\n\twhile true\n\tdo\n\t\tprintf "Use (l)ocal or (r)emote, or (a)bort? "\n\t\tread ans || return 1\n\t\tcase "$ans" in\n\t\t[lL]*)\n\t\t\tif ! local_present\n\t\t\tthen\n\t\t\t\tif test -n "$(git ls-tree HEAD -- "$MERGED")"\n\t\t\t\tthen\n\t\t\t\t\t# Local isn\'t present, but it\'s a subdirectory\n\t\t\t\t\tgit ls-tree --full-name -r HEAD -- "$MERGED" |\n\t\t\t\t\tgit update-index --index-info || exit $?\n\t\t\t\telse\n\t\t\t\t\ttest -e "$MERGED" && mv -- "$MERGED" "$BACKUP"\n\t\t\t\t\tgit update-index --force-remove "$MERGED"\n\t\t\t\t\tcleanup_temp_files --save-backup\n\t\t\t\tfi\n\t\t\telif is_submodule "$local_mode"\n\t\t\tthen\n\t\t\t\tstage_submodule "$MERGED" "$local_sha1"\n\t\t\telse\n\t\t\t\tgit checkout-index -f --stage=2 -- "$MERGED"\n\t\t\t\tgit add -- "$MERGED"\n\t\t\tfi\n\t\t\treturn 0\n\t\t\t;;\n\t\t[rR]*)\n\t\t\tif ! remote_present\n\t\t\tthen\n\t\t\t\tif test -n "$(git ls-tree MERGE_HEAD -- "$MERGED")"\n\t\t\t\tthen\n\t\t\t\t\t# Remote isn\'t present, but it\'s a subdirectory\n\t\t\t\t\tgit ls-tree --full-name -r MERGE_HEAD -- "$MERGED" |\n\t\t\t\t\tgit update-index --index-info || exit $?\n\t\t\t\telse\n\t\t\t\t\ttest -e "$MERGED" && mv -- "$MERGED" "$BACKUP"\n\t\t\t\t\tgit update-index --force-remove "$MERGED"\n\t\t\t\tfi\n\t\t\telif is_submodule "$remote_mode"\n\t\t\tthen\n\t\t\t\t! is_submodule "$local_mode" &&\n\t\t\t\ttest -e "$MERGED" &&\n\t\t\t\tmv -- "$MERGED" "$BACKUP"\n\t\t\t\tstage_submodule "$MERGED" "$remote_sha1"\n\t\t\telse\n\t\t\t\ttest -e "$MERGED" && mv -- "$MERGED" "$BACKUP"\n\t\t\t\tgit checkout-index -f --stage=3 -- "$MERGED"\n\t\t\t\tgit add -- "$MERGED"\n\t\t\tfi\n\t\t\tcleanup_temp_files --save-backup\n\t\t\treturn 0\n\t\t\t;;\n\t\t[aA]*)\n\t\t\treturn 1\n\t\t\t;;\n\t\tesac\n\tdone\n}\n\nstage_submodule () {\n\tpath="$1"\n\tsubmodule_sha1="$2"\n\tmkdir -p "$path" ||\n\tdie "fatal: unable to create directory for module at $path"\n\t# Find $path relative to work tree\n\twork_tree_root=$(cd_to_toplevel && pwd)\n\twork_rel_path=$(cd "$path" &&\n\t\tGIT_WORK_TREE="${work_tree_root}" git rev-parse --show-prefix\n\t)\n\ttest -n "$work_rel_path" ||\n\tdie "fatal: unable to get path of module $path relative to work tree"\n\tgit update-index --add --replace --cacheinfo 160000 "$submodule_sha1" "${work_rel_path%/}" || die\n}\n\ncheckout_staged_file () {\n\ttmpfile="$(git checkout-index --temp --stage="$1" "$2" 2>/dev/null)" &&\n\ttmpfile=${tmpfile%%\'\t\'*}\n\n\tif test $? -eq 0 && test -n "$tmpfile"\n\tthen\n\t\tmv -- "$(git rev-parse --show-cdup)$tmpfile" "$3"\n\telse\n\t\t>"$3"\n\tfi\n}\n\nhide_resolved () {\n\tgit merge-file --ours -q -p "$LOCAL" "$BASE" "$REMOTE" >"$LCONFL"\n\tgit merge-file --theirs -q -p "$LOCAL" "$BASE" "$REMOTE" >"$RCONFL"\n\tmv -- "$LCONFL" "$LOCAL"\n\tmv -- "$RCONFL" "$REMOTE"\n}\n\nmerge_file () {\n\tMERGED="$1"\n\n\tf=$(git ls-files -u -- "$MERGED")\n\tif test -z "$f"\n\tthen\n\t\tif test ! -f "$MERGED"\n\t\tthen\n\t\t\techo "$MERGED: file not found"\n\t\telse\n\t\t\techo "$MERGED: file does not need merging"\n\t\tfi\n\t\treturn 1\n\tfi\n\n\t# extract file extension from the last path component\n\tcase "${MERGED##*/}" in\n\t*.*)\n\t\text=.${MERGED##*.}\n\t\tBASE=${MERGED%"$ext"}\n\t\t;;\n\t*)\n\t\tBASE=$MERGED\n\t\text=\n\tesac\n\n\tinitialize_merge_tool "$merge_tool" || return\n\n\tmergetool_tmpdir_init\n\n\tif test "$MERGETOOL_TMPDIR" != "."\n\tthen\n\t\t# If we\'re using a temporary directory then write to the\n\t\t# top-level of that directory.\n\t\tBASE=${BASE##*/}\n\tfi\n\n\tBACKUP="$MERGETOOL_TMPDIR/${BASE}_BACKUP_$$$ext"\n\tLOCAL="$MERGETOOL_TMPDIR/${BASE}_LOCAL_$$$ext"\n\tLCONFL="$MERGETOOL_TMPDIR/${BASE}_LOCAL_LCONFL_$$$ext"\n\tREMOTE="$MERGETOOL_TMPDIR/${BASE}_REMOTE_$$$ext"\n\tRCONFL="$MERGETOOL_TMPDIR/${BASE}_REMOTE_RCONFL_$$$ext"\n\tBASE="$MERGETOOL_TMPDIR/${BASE}_BASE_$$$ext"\n\n\tbase_mode= local_mode= remote_mode=\n\n\t# here, $IFS is just a LF\n\tfor line in $f\n\tdo\n\t\tmode=${line%% *}\t\t# 1st word\n\t\tsha1=${line#"$mode "}\n\t\tsha1=${sha1%% *}\t\t# 2nd word\n\t\tcase "${line#$mode $sha1 }" in\t# remainder\n\t\t\'1\t\'*)\n\t\t\tbase_mode=$mode\n\t\t\t;;\n\t\t\'2\t\'*)\n\t\t\tlocal_mode=$mode local_sha1=$sha1\n\t\t\t;;\n\t\t\'3\t\'*)\n\t\t\tremote_mode=$mode remote_sha1=$sha1\n\t\t\t;;\n\t\tesac\n\tdone\n\n\tif is_submodule "$local_mode" || is_submodule "$remote_mode"\n\tthen\n\t\techo "Submodule merge conflict for \'$MERGED\':"\n\t\tdescribe_file "$local_mode" "local" "$local_sha1"\n\t\tdescribe_file "$remote_mode" "remote" "$remote_sha1"\n\t\tresolve_submodule_merge\n\t\treturn\n\tfi\n\n\tif test -f "$MERGED"\n\tthen\n\t\tmv -- "$MERGED" "$BACKUP"\n\t\tcp -- "$BACKUP" "$MERGED"\n\tfi\n\t# Create a parent directory to handle delete/delete conflicts\n\t# where the base\'s directory no longer exists.\n\tmkdir -p "$(dirname "$MERGED")"\n\n\tcheckout_staged_file 1 "$MERGED" "$BASE"\n\tcheckout_staged_file 2 "$MERGED" "$LOCAL"\n\tcheckout_staged_file 3 "$MERGED" "$REMOTE"\n\n\t# hideResolved preferences hierarchy.\n\tglobal_config="mergetool.hideResolved"\n\ttool_config="mergetool.${merge_tool}.hideResolved"\n\n\tif enabled=$(git config --type=bool "$tool_config")\n\tthen\n\t\t# The user has a specific preference for a specific tool and no\n\t\t# other preferences should override that.\n\t\t: ;\n\telif enabled=$(git config --type=bool "$global_config")\n\tthen\n\t\t# The user has a general preference for all tools.\n\t\t#\n\t\t# \'true\' means the user likes the feature so we should use it\n\t\t# where possible but tool authors can still override.\n\t\t#\n\t\t# \'false\' means the user doesn\'t like the feature so we should\n\t\t# not use it anywhere.\n\t\tif test "$enabled" = true && hide_resolved_enabled\n\t\tthen\n\t\t    enabled=true\n\t\telse\n\t\t    enabled=false\n\t\tfi\n\telse\n\t\t# The user does not have a preference. Default to disabled.\n\t\tenabled=false\n\tfi\n\n\tif test "$enabled" = true\n\tthen\n\t\thide_resolved\n\tfi\n\n\tif test -z "$local_mode" || test -z "$remote_mode"\n\tthen\n\t\techo "Deleted merge conflict for \'$MERGED\':"\n\t\tdescribe_file "$local_mode" "local" "$LOCAL"\n\t\tdescribe_file "$remote_mode" "remote" "$REMOTE"\n\t\tresolve_deleted_merge\n\t\tstatus=$?\n\t\trmdir -p "$(dirname "$MERGED")" 2>/dev/null\n\t\treturn $status\n\tfi\n\n\tif is_symlink "$local_mode" || is_symlink "$remote_mode"\n\tthen\n\t\techo "Symbolic link merge conflict for \'$MERGED\':"\n\t\tdescribe_file "$local_mode" "local" "$LOCAL"\n\t\tdescribe_file "$remote_mode" "remote" "$REMOTE"\n\t\tresolve_symlink_merge\n\t\treturn\n\tfi\n\n\techo "Normal merge conflict for \'$MERGED\':"\n\tdescribe_file "$local_mode" "local" "$LOCAL"\n\tdescribe_file "$remote_mode" "remote" "$REMOTE"\n\tif test "$guessed_merge_tool" = true || test "$prompt" = true\n\tthen\n\t\tprintf "Hit return to start merge resolution tool (%s): " "$merge_tool"\n\t\tread ans || return 1\n\tfi\n\n\tif base_present\n\tthen',
      repo: "git",
    },
  ],
  time_taken: 0.8760850429534912,
};

const aiexplanationapiResponse = `
Sure! Let's break down the code in simple terms:

\`\`\`python
def add(a, b):
   return a + b
\`\`\`

1. **Function Definition**: The line \`def add(a, b):\` is defining a function named \`add\`. A function is like a small block of code that does a specific task. In this case, the task is to add two numbers.

2. **Parameters**: The function \`add\` takes two parameters, \`a\` and \`b\`. Parameters are like placeholders for the values that you will give to the function when you use it. Here, \`a\` and \`b\` will be the numbers you want to add together.

3. **Return Statement**: The line \`return a + b\` tells the function what to do with the parameters. In this case, it adds \`a\` and \`b\` together using the \`+\` operator, which is used for addition in Python. The \`return\` keyword then sends the result of this addition back to wherever the function was called.

4. **Using the Function**: To use this function, you would call it with two numbers, like \`add(3, 5)\`. This would return \`8\` because \`3 + 5\` equals \`8\`.

So, in simple terms, this code creates a function that takes two numbers as input and returns their sum.
`;
const airefactorapiResponse = `
The function \`add\` is already quite simple and straightforward, so there isn't much to refactor in terms of functionality or performance. However, there are a few suggestions that can improve the code's readability, maintainability, and robustness:

1. **Type Hinting**: Adding type hints can make the code more understandable and help with static analysis tools.

2. **Docstring**: Including a docstring can provide clarity on what the function does, which is especially useful for more complex functions.

3. **Validation**: Depending on the context in which this function is used, you might want to add some validation to ensure that the inputs are of the expected types.

Here's how you might refactor the function with these suggestions:

\`\`\`python
def add(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.

    Parameters:
    a (float): The first number.
    b (float): The second number.

    Returns:
    float: The sum of the two numbers.
    """
    # Validate input types
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers (int or float).")
    
    return a + b
\`\`\`

These changes make the function more robust and easier to understand, especially in larger codebases.
`;

const aisecurityscanapiResponse = `
The provided function \`add(a, b)\` is quite simple and, from a security perspective, does not appear to have any inherent vulnerabilities such as SQL injection, cross-site scripting (XSS), buffer overflows, or other common security issues. However, there are a few considerations and best practices that can be applied to ensure security and compliance:

### Security Considerations

1. **Input Validation and Type Checking**:
  - The function does not perform any validation on the inputs \`a\` and \`b\`. If these inputs are coming from an untrusted source, it could lead to unexpected behavior or errors.
  - Consider adding type checking to ensure that \`a\` and \`b\` are of the expected type (e.g., integers or floats).

2. **Error Handling**:
  - While the addition operation itself is simple, consider adding error handling to manage potential exceptions (e.g., if the inputs are not numbers).

3. **Code Injection**:
  - Although unlikely in this simple function, ensure that the inputs are not being used in a way that could lead to code injection, especially if this function is part of a larger system where inputs are dynamically executed.

### Compliance Considerations

1. **Data Privacy**:
  - If \`a\` and \`b\` represent sensitive data, ensure that the function and any surrounding code comply with data privacy regulations such as GDPR, HIPAA, etc.
  - Consider anonymizing or encrypting sensitive data before processing.

2. **Code Review and Security Audits**:
  - Regular code reviews and security audits should be performed to identify and mitigate potential security issues.

3. **Documentation and Comments**:
  - Ensure that the function and any related code are well-documented, including assumptions about input types and expected behavior.

4. **Access Control**:
  - Ensure that only authorized users or systems can access and modify the code or the data being processed.

### Example of Enhanced Function

Here is an example of how you might enhance the function with some of these considerations:

\`\`\`python
def add(a, b):
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
   
   return a + b
\`\`\`

This version of the function includes type checking and a docstring, which can help with both security and compliance by making the function's behavior and requirements clear.
`;

const page = () => {
  return (
    <div className="relative w-[65rem] mx-auto overflow-hidden">
      {/* Prompt bubble */}
      <p className="p-3 my-5 w-fit max-w-2xl bg-neutral-300 dark:bg-neutral-700 rounded-b-2xl rounded-tl-2xl ml-auto">
        Where does Git detect and handle merge conflicts?
      </p>

      {/* Scrollable horizontal area */}
      <div className="h-[calc(100%-5rem)] overflow-x-auto scrollbar-thumb-neutral-500 scrollbar-track-transparent">
        <div className="flex w-max min-w-full">
          {/* Code block section */}
          <div className="flex flex-col pl-0 pr-2 p-4 min-w-lg">
            {results.results.map((item, index) => {
              const fileExtension = item.filename.includes(".")
                ? item.filename.split(".").pop()
                : "none";
              return (
                <div
                  key={index}
                  className="mb-6 rounded-3xl bg-neutral-200 dark:bg-neutral-900 p-6 max-w-lg"
                >
                  <p className="mb-2 text-sm">
                    <strong>Filepath:</strong> {item.filepath}
                  </p>
                  <CodeBlock
                    language={fileExtension ?? ""}
                    filename={item.filepath}
                    code={item.content}
                  />
                </div>
              );
            })}
          </div>

          {/* AI section */}
          <div className="flex flex-col pr-0 pl-2 p-4 min-w-lg space-y-4">
            <LLMResponse title="AI Explanation" content={aiexplanationapiResponse} />
            <LLMResponse title="AI Refactor Suggestion" content={airefactorapiResponse} />
            <LLMResponse title="AI Security Scan" content={aisecurityscanapiResponse} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default page;

