import sys
import pandas as pd
from io import StringIO
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget
from PyQt6.uic import loadUi

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = loadUi("./PPFML.ui", self)
        self.show()

        #Button events
        self.ui.info_btn.clicked.connect(self.displayDataInfo)

    def qtableToDataFrame(self, QTable : QTableWidget) -> pd.DataFrame:
        """Convert a QTableWidget to a pandas DataFrame
        This method extracts the text data from a QTableWidget and converts it into
        a pandas DataFrame. Each column in the QTableWidget is converted to a column
        in the DataFrame. The header of the QTableWidget, if set, is used to name
        the DataFrame's columns. If no header is present, columns are named using
        their index values.

        Parameters:
        ------------
        *QTable (Mandatory, QTableWidget)*: The QTableWidget instance to convert.

        Returns:
        ------------
        *pd.DataFrame*: A pandas DataFrame containing the data from the QTableWidget.

        Note:
        - If a cell in the QTableWidget is empty or unset, it will be represented as None in the DataFrame.
        - This function assumes that the QTableWidget is fully populated, i.e., every row and column index
        that is accessed is within the bounds of the QTableWidget's dimensions.
        - This method does not handle data types. All data is extracted as strings and it is the
        responsibility of the caller to convert the data to the correct type if necessary.
        """
        data = {}
        rows = QTable.rowCount()
        col = QTable.columnCount()
        
        for c in range(col):
            col_data=[]
            header = QTable.horizontalHeaderItem(c)
            col_name = header.text() if header is not None else str(c)
            for r in range(rows):
                item = QTable.item(r,c)
                col_data.append(item.text() if item is not None else None)
            
            data[col_name] = col_data
        
        df = pd.DataFrame(data)
        return df

    def displayDataInfo(self):
        """Retrieves data from a QTableWidget, converts it to a pandas DataFrame, and displays
            the information summary of the DataFrame in a QTextEdit widget.

        This method uses the `qtableToDataFrame` method to convert the QTableWidget's contents
        into a pandas DataFrame. It then uses the DataFrame's `info()` method to get a summary
        of the data, which includes the index dtype, column dtypes, non-null values, and memory usage.
        The summary is captured in a StringIO buffer and then set as the text of the QTextEdit widget
        for display.

        The QTextEdit widget used for display is assumed to be named `infoTextEdit` and should be
        part of the UI elements accessible through the `ui` attribute of the class instance.

        Note:
        - This method will overwrite the current text in the `infoTextEdit` widget.
        - The DataFrame's `info()` method output is purely informational and is best used
        for quick data assessment and debugging purposes.
        - It is assumed that `self.ui.tableWidget` is the QTableWidget instance from which data
        will be extracted and `self.ui.infoTextEdit` is the QTextEdit instance where the data
        info will be displayed.
        """
        buffer = StringIO()
        data = self.qtableToDataFrame(self.ui.tableWidget)
        data.info(buf=buffer)
        info = buffer.getvalue()
        self.ui.infoTextEdit.setText(info)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec())
