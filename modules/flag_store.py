import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from datetime import datetime

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1FvMb4GZi2T1g6EN4uLMew1RVS3ua0BUbPOPfRSiO8Hc"


class FlagStore:
    def __init__(self):
        self._sheet = None

    def _connect(self):
        if self._sheet is None:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=SCOPES
            )
            client = gspread.authorize(creds)
            self._sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        return self._sheet

    def append_flag(self, data: dict):
        try:
            sheet = self._connect()
            # Add headers if sheet is empty
            if not sheet.get_all_values():
                sheet.append_row([
                    "timestamp", "food", "normalized", "bls302_code",
                    "bls302_name", "bls40_code", "bls40_name", "nova",
                    "source", "note",
                ])
            row = [
                data.get("timestamp", datetime.now().isoformat()),
                data.get("food", ""),
                data.get("normalized", ""),
                data.get("bls302_code", ""),
                data.get("bls302_name", ""),
                data.get("bls40_code", ""),
                data.get("bls40_name", ""),
                str(data.get("nova", "")),
                data.get("source", ""),
                data.get("note", ""),
            ]
            sheet.append_row(row)
            return True
        except Exception as e:
            print(f"Flag store error: {e}")
            return False

    def get_all_flags(self):
        try:
            sheet = self._connect()
            return sheet.get_all_records()
        except Exception as e:
            print(f"Flag read error: {e}")
            return []
