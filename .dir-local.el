((python-mode . ((eval . (setq-local compile-command
				     (concat "uv run python " (or (buffer-file-name) "")))))))
