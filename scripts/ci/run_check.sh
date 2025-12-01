#!/bin/bash
set -e

run_check_with_summary() {
    local check_name="$1"
    local icon="$2"
    local title="$3"
    local make_command="$4"
    local output_file="$5"
    local success_message="$6"
    local error_message="$7"
    local fix_suggestion="$8"
    local output_format="${9:-text}"  # Default to 'text', can be 'diff'

    echo "## $icon $title" >> $GITHUB_STEP_SUMMARY

    if $make_command > "$output_file" 2>&1; then
        echo "✅ **$success_message**" >> $GITHUB_STEP_SUMMARY
    else
        echo "❌ **$error_message**" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "### Errors found:" >> $GITHUB_STEP_SUMMARY

        if [ "$output_format" = "diff" ]; then
            echo '```diff' >> $GITHUB_STEP_SUMMARY
        else
            echo '```' >> $GITHUB_STEP_SUMMARY
        fi

        cat "$output_file" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "💡 **Fix suggestion:** $fix_suggestion" >> $GITHUB_STEP_SUMMARY
        exit 1
    fi
}

run_test_check() {
    echo "## 🧪 Test Results" >> $GITHUB_STEP_SUMMARY

    if make test > test_output.txt 2>&1; then
        echo "✅ **All tests passed!**" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY

        # Extract test summary
        if grep -q "=" test_output.txt; then
            echo "### Test Summary:" >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
            grep -E "(passed|failed|error|skipped|=)" test_output.txt | tail -5 >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
        fi

        # Extract coverage summary
        if grep -q "TOTAL" test_output.txt; then
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "### Coverage Report:" >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
            grep -A 10 "Name.*Stmts.*Miss.*Cover" test_output.txt >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
        fi
    else
        echo "❌ **Tests failed!**" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "### Test output:" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        cat test_output.txt >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "💡 **Fix suggestion:** Review failing tests and fix issues" >> $GITHUB_STEP_SUMMARY
        exit 1
    fi
}

run_version_check() {
    echo "## 🔢 Version Validation" >> $GITHUB_STEP_SUMMARY

    if make validate-version > version_output.txt 2>&1; then
        echo "✅ **Version validation passed!**" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "### Validation result:" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        cat version_output.txt >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
    else
        echo "❌ **Version validation failed!**" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "### Version mismatch:" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        cat version_output.txt >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "💡 **Fix suggestion:** Update version in pyproject.toml or __init__.py to match" >> $GITHUB_STEP_SUMMARY
        exit 1
    fi
}

case "$1" in
    lint)
        run_check_with_summary \
            "lint" \
            "🔍" \
            "Linting Results" \
            "make lint" \
            "lint_output.txt" \
            "All linting checks passed!" \
            "Linting failed!" \
            "Run \`make lint-fix\` to auto-fix issues"
        ;;
    format-check)
        run_check_with_summary \
            "format-check" \
            "🎨" \
            "Code Formatting Check" \
            "make format-check" \
            "format_output.txt" \
            "Code is properly formatted!" \
            "Formatting issues found!" \
            "Run \`make format\` to auto-format code" \
            "diff"
        ;;
    type-check)
        run_check_with_summary \
            "type-check" \
            "🔎" \
            "Type Checking Results" \
            "make type-check" \
            "mypy_output.txt" \
            "All type checks passed!" \
            "Type checking failed!" \
            "Add type hints and fix type errors"
        ;;
    test)
        run_test_check
        ;;
    validate-version)
        run_version_check
        ;;
    *)
        echo "Unknown check type: $1"
        exit 1
        ;;
esac
