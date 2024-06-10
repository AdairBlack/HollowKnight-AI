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

        Parameters:

        Returns:
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
