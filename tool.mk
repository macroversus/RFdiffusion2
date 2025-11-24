
TOOL_NAME := RFdiffusion2
TOOL_VERSION := 0.0.1
TOOL_DIR := $(shell pwd)

UV_HTTP_TIMEOUT := 600

.PHONY: install
install: 
	@pixi run setup
	@echo "$(COLOR_GREEN)✓ RFdiffusion2 安装完成$(COLOR_RESET)"
