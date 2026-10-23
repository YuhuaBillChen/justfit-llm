# Modified for JustFit: product identity and integration.
# Keep the stable install path so an already-running shell survives upgrades.
set -g __justfit_completion_source (builtin realpath -s -- (status filename))

function __justfit_models
    # The helper beside the script's current target, found again each time.
    set -l helper (path dirname -- (path resolve -- $__justfit_completion_source))/models
    test -x $helper
    and $helper
end

# As in Bash and Zsh: only the value of --model, and only after `justfit serve`.
function __justfit_model_value
    set -l words (commandline -opc)
    test "$words[2]" = serve
    and not string match -qr -- '^-[^=]*$' (commandline -ct)
end

complete -c justfit -f
complete -c justfit -n 'test (count (commandline -opc)) -eq 1' -a 'serve claude codex opencode hermes pi'
complete -c justfit -n __justfit_model_value -l model -x -a '(__justfit_models)'
