class observer():
    _handlers:list[callable] = []

    def __init__(self):
        self._handlers = []

    def connect(self, handler:callable):
        if handler not in self._handlers:
            self._handlers.append(handler)
            print(f"핸들러 '{handler.__name__}' 등록 완료.")

    def _notify_handlers(self, new_value):
        """등록된 모든 핸들러 함수를 호출합니다."""
        for handler in self._handlers:
            handler(new_value)