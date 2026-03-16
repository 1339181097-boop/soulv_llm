from pipeline.data_utils import configure_console_output, log_info


def main() -> None:
    configure_console_output()
    log_info("搜旅 LLM 项目已初始化，当前重点是 Phase 1 数据管道建设。")
    log_info("本地环境负责数据清洗，阿里云环境负责训练、推理和部署。")


if __name__ == "__main__":
    main()
