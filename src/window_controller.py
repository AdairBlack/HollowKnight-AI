import pyautogui


class WindowController():
    """
    Class to control the window of the application.
    """

    def __init__(self):
        self.monitor = self._find_window()

    def _find_window():
        """
        find the location of Hollow Knight window

        window: the location of the window
        """

        windows = pyautogui.getWindowsWithTitle('Hollow Knight')
        if len(windows) == 0:
            raise ValueError('Hollow Knight window not found')
        if len(windows) > 1:
            raise ValueError(
                f'Multiple Hollow Knight windows found, count: {len(windows)}')

        window = windows[0]

        return window

    def _find_menu(self):
        """
        locate the menu badge,
        when the badge is found, the correct game is ready to be started

        :return: the location of menu badge
        """
        monitor = self.monitor
        monitor = (monitor['left'] + monitor['width'] // 2,
                   monitor['top'] + monitor['height'] // 4,
                   monitor['width'] // 2,
                   monitor['height'] // 2)
        return pyautogui.locateOnScreen(f'locator/menu_badge.png',
                                        region=monitor,
                                        confidence=0.925)
