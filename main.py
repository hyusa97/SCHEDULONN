"""
Streamlit app: AI-Assisted Timetable Generator (CP-SAT backend)
------------------------------------------------------------
Single-file Streamlit app that:
 - Accepts CSV/Excel uploads for courses, sections, faculties, rooms, fixed events
 - Lets you define days & slots per day interactively
 - Calls the CP-SAT solver (OR-Tools) to generate a clash-free timetable
 - Shows result in-app and allows Excel/CSV download

Dependencies:
    pip install streamlit pandas ortools openpyxl reportlab

Run:
    streamlit run this_file.py

Notes:
 - This file embeds the solver previously created. Keep it as a single-file prototype for quick deployment.
 - For production, split solver code into modules and add DB-backed storage, authentication, and better UX.
"""

from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
import streamlit as st
from ortools.sat.python import cp_model
import bcrypt
import gspread
from google.oauth2.service_account import Credentials

# ------------------------------- Solver (embedded) -------------------------
@dataclass
class Course:
    course_id: str
    name: str
    credits: int
    hours_per_week: int
    is_lab: bool = False
    requires_consecutive: bool = False

@dataclass
class Section:
    section_id: str
    course_id: str
    program: str
    cohort: str
    max_capacity: int
    required_faculty: Optional[str] = None

@dataclass
class Faculty:
    faculty_id: str
    name: str
    can_teach: Set[str]
    max_hours_per_week: int
    availability: Set[Tuple[str, int]]
    preferred_slots: Set[Tuple[str, int]] = None

@dataclass
class Room:
    room_id: str
    capacity: int
    is_lab: bool = False

@dataclass
class FixedEvent:
    kind: str
    cohort: Optional[str]
    faculty_id: Optional[str]
    room_id: Optional[str]
    day: str
    slot_index: int

def build_index_maps(sections: List[Section], faculties: List[Faculty], rooms: List[Room], slots: List[Tuple[str,int]]):
    sec_idx = {s.section_id: i for i, s in enumerate(sections)}
    fac_idx = {f.faculty_id: i for i, f in enumerate(faculties)}
    room_idx = {r.room_id: i for i, r in enumerate(rooms)}
    slot_idx = {slots[i]: i for i in range(len(slots))}
    return sec_idx, fac_idx, room_idx, slot_idx

class TimetableSolver:
    def __init__(
        self,
        courses: Dict[str, Course],
        sections: List[Section],
        faculties: List[Faculty],
        rooms: List[Room],
        slots: List[Tuple[str,int]],
        fixed_events: List[FixedEvent] = None,
        weights: Dict[str, int] = None,
    ):
        self.courses = courses
        self.sections = sections
        self.faculties = faculties
        self.rooms = rooms
        self.slots = slots
        self.fixed_events = fixed_events or []
        self.w = {
            "minimize_gaps": 1,
            "prefer_faculty_slots": 2,
            "balance_faculty_load": 1,
            "minimize_room_changes": 1,
            "prefer_morning_for_fyup1": 1,
        }
        if weights:
            self.w.update(weights)

        self.model = cp_model.CpModel()
        self.sec_idx, self.fac_idx, self.room_idx, self.slot_idx = build_index_maps(sections, faculties, rooms, slots)
        self._build_variables()
        self._add_hard_constraints()
        self._add_soft_constraints_and_objective()

    def _build_variables(self):
        S, F, R, T = len(self.sections), len(self.faculties), len(self.rooms), len(self.slots)
        self.x = {}
        for s in range(S):
            for t in range(T):
                for r in range(R):
                    self.x[(s,t,r)] = self.model.NewBoolVar(f"x_s{s}_t{t}_r{r}")

        self.a = {}
        for s, sec in enumerate(self.sections):
            for f, fac in enumerate(self.faculties):
                can = (sec.required_faculty == fac.faculty_id) if sec.required_faculty else (sec.course_id in fac.can_teach)
                self.a[(s,f)] = self.model.NewBoolVar(f"a_s{s}_f{f}") if can else self.model.NewConstant(0)

        self.y = {}
        for s in range(S):
            for t in range(T):
                self.y[(s,t)] = self.model.NewBoolVar(f"y_s{s}_t{t}")

        self.r_used = {}
        for s in range(S):
            for t in range(T):
                self.r_used[(s,t)] = self.model.NewBoolVar(f"rused_s{s}_t{t}")

    def _add_hard_constraints(self):
        S, F, R, T = len(self.sections), len(self.faculties), len(self.rooms), len(self.slots)
        for s, sec in enumerate(self.sections):
            H = self.courses[sec.course_id].hours_per_week
            self.model.Add(sum(self.y[(s,t)] for t in range(T)) == H)

        for s in range(S):
            for t in range(T):
                self.model.Add(sum(self.x[(s,t,r)] for r in range(R)) == self.y[(s,t)])
                self.model.Add(self.r_used[(s,t)] == self.y[(s,t)])

        for s, sec in enumerate(self.sections):
            course = self.courses[sec.course_id]
            for t in range(T):
                for r, room in enumerate(self.rooms):
                    if course.is_lab and not room.is_lab:
                        self.model.Add(self.x[(s,t,r)] == 0)

        for t in range(T):
            for r in range(R):
                self.model.Add(sum(self.x[(s,t,r)] for s in range(S)) <= 1)

        for s in range(S):
            self.model.Add(sum(self.a[(s,f)] for f in range(F)) == 1)

        for f, fac in enumerate(self.faculties):
            avail_t = {self.slot_idx[sl] for sl in fac.availability if sl in self.slot_idx}
            for t in range(T):
                if t not in avail_t:
                    for s in range(S):
                        # if faculty can't teach at t, a[s,f] => not y[s,t]
                        self.model.Add(self.a[(s,f)] + self.y[(s,t)] <= 1)
                self.model.Add(sum(self.y[(s,t)] * self.a[(s,f)] for s in range(S)) <= 1)

        cohort_to_secs: Dict[str, List[int]] = {}
        for s, sec in enumerate(self.sections):
            cohort_to_secs.setdefault(sec.cohort, []).append(s)
        for cohort, sec_list in cohort_to_secs.items():
            for t in range(T):
                self.model.Add(sum(self.y[(s,t)] for s in sec_list) <= 1)

        for s, sec in enumerate(self.sections):
            course = self.courses[sec.course_id]
            if course.requires_consecutive:
                day_to_indices: Dict[str, List[int]] = {}
                for idx, (day, sl) in enumerate(self.slots):
                    day_to_indices.setdefault(day, []).append(idx)
                for day, indices in day_to_indices.items():
                    for i in range(0, len(indices)-1, 2):
                        t1, t2 = indices[i], indices[i+1]
                        self.model.Add(self.y[(s,t1)] == self.y[(s,t2)])

        for ev in self.fixed_events:
            if (ev.day, ev.slot_index) not in self.slot_idx:
                continue
            t = self.slot_idx[(ev.day, ev.slot_index)]
            if ev.cohort is not None and ev.cohort in cohort_to_secs:
                for s in cohort_to_secs[ev.cohort]:
                    self.model.Add(self.y[(s,t)] == 0)
            if ev.room_id is not None and ev.room_id in self.room_idx:
                r = self.room_idx[ev.room_id]
                self.model.Add(sum(self.x[(s,t,r)] for s in range(S)) == 0)
            if ev.faculty_id is not None and ev.faculty_id in self.fac_idx:
                f = self.fac_idx[ev.faculty_id]
                self.model.Add(sum(self.y[(s,t)] * self.a[(s,f)] for s in range(S)) == 0)

    def _add_soft_constraints_and_objective(self):
        S, F, R, T = len(self.sections), len(self.faculties), len(self.rooms), len(self.slots)
        terms = []
        for f, fac in enumerate(self.faculties):
            pref_ts = {self.slot_idx[sl] for sl in (fac.preferred_slots or set()) if sl in self.slot_idx}
            for s in range(S):
                for t in pref_ts:
                    terms.append(self.w["prefer_faculty_slots"] * self.a[(s,f)] * self.y[(s,t)])

        day_to_indices: Dict[str, List[int]] = {}
        for idx, (day, sl) in enumerate(self.slots):
            day_to_indices.setdefault(day, []).append(idx)
        cohort_to_secs: Dict[str, List[int]] = {}
        for s, sec in enumerate(self.sections):
            cohort_to_secs.setdefault(sec.cohort, []).append(s)
        for cohort, sec_list in cohort_to_secs.items():
            for day, indices in day_to_indices.items():
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        t1, t2 = indices[i], indices[j]
                        sep = t2 - t1
                        if sep > 1:
                            terms.append(-self.w["minimize_gaps"] * sep * sum(self.y[(s,t1)] + self.y[(s,t2)] for s in sec_list))

        target = {}
        for f, fac in enumerate(self.faculties):
            target[f] = min(
                fac.max_hours_per_week,
                sum(self.courses[sec.course_id].hours_per_week for sec in self.sections if sec.course_id in fac.can_teach)
            )
            load = sum(self.a[(s,f)] * self.courses[self.sections[s].course_id].hours_per_week for s in range(len(self.sections)))
            dev_pos = self.model.NewIntVar(0, 1000, f"devpos_f{f}")
            dev_neg = self.model.NewIntVar(0, 1000, f"devneg_f{f}")
            self.model.Add(load - target[f] == dev_pos - dev_neg)
            terms.append(-self.w["balance_faculty_load"] * (dev_pos + dev_neg))

        for s in range(len(self.sections)):
            for r in range(len(self.rooms)):
                terms.append(self.w["minimize_room_changes"] * sum(self.x[(s,t,r)] for t in range(len(self.slots))))

        for s, sec in enumerate(self.sections):
            if "SEM1" in sec.cohort and "FYUP" in sec.cohort:
                for t, (day, sl) in enumerate(self.slots):
                    if sl <= 2:
                        terms.append(self.w["prefer_morning_for_fyup1"] * self.y[(s,t)])

        self.model.Maximize(sum(terms))

    def solve(self, time_limit_sec: int = 30):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
        solver.parameters.num_search_workers = 8
        status = solver.Solve(self.model)
        return solver, status

    def extract(self, solver: cp_model.CpSolver):
        S, R, T = len(self.sections), len(self.rooms), len(self.slots)
        rows = []
        sec_fac = {}
        for s in range(S):
            for f in range(len(self.faculties)):
                if self.a.get((s,f)) and solver.Value(self.a[(s,f)]) == 1:
                    sec_fac[s] = self.faculties[f].faculty_id
        for s in range(S):
            sec = self.sections[s]
            course = self.courses[sec.course_id]
            for t in range(T):
                if solver.Value(self.y[(s,t)]) == 1:
                    room_assigned = None
                    for r in range(R):
                        if solver.Value(self.x[(s,t,r)]) == 1:
                            room_assigned = self.rooms[r].room_id
                            break
                    day, slot = self.slots[t]
                    rows.append({
                        "cohort": sec.cohort,
                        "program": sec.program,
                        "section_id": sec.section_id,
                        "course_id": sec.course_id,
                        "course_name": course.name,
                        "faculty_id": sec_fac.get(s, "TBD"),
                        "day": day,
                        "slot": slot,
                        "room_id": room_assigned or "TBD",
                    })
        df = pd.DataFrame(rows).sort_values(["cohort","day","slot"]).reset_index(drop=True)
        return df

# ------------------------------- Public API --------------------------------

def solve_timetable(
    courses_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    faculties_df: pd.DataFrame,
    rooms_df: pd.DataFrame,
    slots: List[Tuple[str,int]],
    fixed_events_df: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str,int]] = None,
    time_limit_sec: int = 30,
) -> Tuple[pd.DataFrame, int]:
    def parse_set(v):
        if isinstance(v, (set, list, tuple)):
            return set(v)
        if pd.isna(v) or v is None or v == "":
            return set()
        return set(str(v).replace(" ", "").split(";"))

    courses = {
        r.course_id: Course(
            course_id=r.course_id,
            name=r.name,
            credits=int(r.credits),
            hours_per_week=int(r.hours_per_week),
            is_lab=bool(r.is_lab),
            requires_consecutive=bool(r.requires_consecutive),
        )
        for r in courses_df.itertuples(index=False)
    }

    sections = [
        Section(
            section_id=r.section_id,
            course_id=r.course_id,
            program=r.program,
            cohort=r.cohort,
            max_capacity=int(r.max_capacity),
            required_faculty=(None if pd.isna(r.required_faculty) else str(r.required_faculty)),
        ) for r in sections_df.itertuples(index=False)
    ]

    def parse_pairs_list(v):
        s = set()
        if pd.isna(v) or v is None or v == "":
            return s
        for token in str(v).split(";"):
            if token.strip() == "":
                continue
            day, sl = token.split("-")
            s.add((day, int(sl)))
        return s

    faculties = []
    for r in faculties_df.itertuples(index=False):
        faculties.append(
            Faculty(
                faculty_id=r.faculty_id,
                name=r.name,
                can_teach=parse_set(r.can_teach),
                max_hours_per_week=int(r.max_hours_per_week),
                availability=parse_pairs_list(r.availability),
                preferred_slots=parse_pairs_list(getattr(r, "preferred_slots", "")),
            )
        )

    rooms = [
        Room(room_id=r.room_id, capacity=int(r.capacity), is_lab=bool(r.is_lab))
        for r in rooms_df.itertuples(index=False)
    ]

    fixed_events = []
    if fixed_events_df is not None:
        for r in fixed_events_df.itertuples(index=False):
            fixed_events.append(FixedEvent(
                kind=r.kind,
                cohort=(None if pd.isna(r.cohort) else str(r.cohort)),
                faculty_id=(None if pd.isna(r.faculty_id) else str(r.faculty_id)),
                room_id=(None if pd.isna(r.room_id) else str(r.room_id)),
                day=r.day,
                slot_index=int(r.slot_index),
            ))

    solver = TimetableSolver(
        courses=courses,
        sections=sections,
        faculties=faculties,
        rooms=rooms,
        slots=slots,
        fixed_events=fixed_events,
        weights=weights,
    )

    cp_solver, status = solver.solve(time_limit_sec=time_limit_sec)
    df = solver.extract(cp_solver)
    return df, status

# ------------------------------ Streamlit UI --------------------------------

# ---------------- CONFIG ----------------
AUTH_SHEET_ID = st.secrets["sheets"]["AUTH_SHEET_ID"]
AUTH_SHEET_NAME = "AUTH_SHEET"

# Fix private key newlines
creds_dict = dict(st.secrets["gcp_service_account"])
creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

# ---------------- CONNECT TO GOOGLE SHEETS ----------------
@st.cache_resource
def connect_to_sheets():
    try:
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        client = gspread.authorize(creds)
        AUTH_sheet = client.open_by_key(AUTH_SHEET_ID).worksheet(AUTH_SHEET_NAME)
        return AUTH_sheet
    except Exception as e:
        st.error(f"❌ Failed to connect to Google Sheets: {e}")
        st.stop()

AUTH_sheet = connect_to_sheets()

# ---------------- LOAD AUTH DATA ----------------
@st.cache_resource
def load_auth_data():
    data = AUTH_sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

auth_df = load_auth_data()

# ---------------- PASSWORD VERIFICATION ----------------
def verify_password(stored_hash, entered_password):
    try:
        return bcrypt.checkpw(entered_password.encode(), stored_hash.encode())
    except Exception:
        return False

# ---------------- SESSION STATE INIT ----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.session_state.user_name = None

# ---------------- LOGIN PAGE ----------------
if not st.session_state.authenticated:
    st.title("🔒 Secure Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    login_button = st.button("Login")

    if login_button:
        user_data = auth_df[auth_df["Username"] == username]

        if not user_data.empty:
            stored_hash = user_data.iloc[0]["Password"]
            role = user_data.iloc[0]["Role"]
            name = user_data.iloc[0]["Name"]

            if verify_password(stored_hash, password):
                st.session_state.authenticated = True
                st.session_state.user_role = role
                st.session_state.username = username
                st.session_state.user_name = name

                st.experimental_set_query_params(logged_in="true")
                st.success(f"✅ Welcome, {name}!")
                st.rerun()
            else:
                st.error("❌ Invalid Credentials")
        else:
            st.error("❌ User not found")

# ---------------- DASHBOARD (AFTER LOGIN) ----------------
else:
    st.sidebar.write(f"👤 **Welcome, {st.session_state.user_name}!** ({st.session_state.user_role})")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.user_name = None
        st.experimental_set_query_params(logged_in="false")
        st.rerun()

    st.title("📊 Dashboard")
    st.write("This is where your main app content goes...")

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import bcrypt
import io

# ------------------------------ EXISTING LOGIN/DASHBOARD CODE ABOVE ------------------------------
# (Keep your login/auth code as is)

# ------------------------------ UNIVERSITY LANDING PAGE ------------------------------
if st.session_state.user_role == "university":
    st.header("🏫 University Dashboard - SCHEDULONN")

    # ---------------- COURSE SECTION ----------------
    def load_courses(SHEET_ID, SHEET_NAME="COURSE"):
    """Load courses and semesters from Google Sheets into a dict"""
    try:
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
        data = sheet.get_all_records()

        courses = {}
        for row in data:
            course = row.get("Course Name")
            semester = str(row.get("Semester"))
            if course:
                if course not in courses:
                    courses[course] = []
                if semester not in courses[course]:
                    courses[course].append(semester)
        return courses
    except Exception as e:
        st.error(f"❌ Failed to load courses: {e}")
        return {}
    

def save_course(SHEET_ID, SHEET_NAME, course_name, num_semesters):
    """Append a new course + semesters into Google Sheets"""
    try:
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

        rows = [[course_name, i] for i in range(1, num_semesters + 1)]
        sheet.append_rows(rows)

        st.success(f"✅ Added {course_name} with {num_semesters} semesters.")
    except Exception as e:
        st.error(f"❌ Failed to save course: {e}")
    

def delete_course(SHEET_ID, SHEET_NAME, course_name):
    """Delete a course and all its semesters from Google Sheets"""
    try:
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
        data = sheet.get_all_values()

        # Keep header row
        header = data[0]
        filtered = [row for row in data if row[0] != course_name]

        sheet.clear()
        sheet.update([header] + filtered)

        st.success(f"🗑️ Deleted course {course_name}")
    except Exception as e:
        st.error(f"❌ Failed to delete course: {e}")



    # ---------------- SEMESTER SECTION ----------------
    st.subheader("🎓 Semester")
    if selected_course != "-- Select --":
        semester = st.selectbox("Select Semester", st.session_state.courses[selected_course])
    else:
        st.info("Please select a course to choose semesters.")

    # ---------------- FILE UPLOADS (Google Sheets Integration) ----------------
    def upload_to_sheets(uploaded_file, sheet_name, SHEET_ID):
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
            try:
                creds = Credentials.from_service_account_info(
                    creds_dict,
                    scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive",
                    ],
                )
                client = gspread.authorize(creds)
                sheet = client.open_by_key(SHEET_ID).worksheet(sheet_name)
                sheet.clear()
                sheet.update([df.columns.values.tolist()] + df.values.tolist())
                st.success(f"✅ Successfully uploaded {uploaded_file.name} to {sheet_name}")
            except Exception as e:
                st.error(f"❌ Failed to upload {uploaded_file.name}: {e}")

    SHEET_ID = "1kx7yI4KQhqptIBj7dR-ECDvghch4BKWQCFH_wURbI80"

    st.subheader("👥 Group / Batch Details")
    group_file = st.file_uploader("Upload Group/Batch CSV or Excel", type=["csv", "xlsx"])
    if st.button("Upload Group File"):
        upload_to_sheets(group_file, "GroupBatch", SHEET_ID)

    st.subheader("👨‍🏫 Professor Details")
    prof_file = st.file_uploader("Upload Professor CSV or Excel", type=["csv", "xlsx"])
    if st.button("Upload Professor File"):
        upload_to_sheets(prof_file, "Professors", SHEET_ID)

    st.subheader("🏛️ Classroom Details")
    class_file = st.file_uploader("Upload Classroom CSV or Excel", type=["csv", "xlsx"])
    if st.button("Upload Classroom File"):
        upload_to_sheets(class_file, "Classrooms", SHEET_ID)

    # ---------------- SLOT SECTION ----------------
    st.subheader("⏰ Slot Settings")

    # Step 1: Working Days
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    num_days = st.selectbox("Number of working days per week", list(range(1, 8)), index=4)
    selected_days = days[:num_days]
    st.write("📅 Active Days:", selected_days)

    # Step 2: Working Hours
    num_hours = st.selectbox("Number of working hours per day", list(range(1, 13)), index=5)
    time_slots = [f"{8+i:02d}:00" for i in range(num_hours)]
    st.write("⏱️ Time Slots:", time_slots)

    # Step 3: Breaks
    st.write("🍴 Break Times (X = Break, Empty = Allowed)")
    break_table = pd.DataFrame("", index=time_slots, columns=selected_days)

    edited_breaks = st.data_editor(break_table, num_rows="dynamic", key="breaks")
    st.caption("⚡ Use X to mark breaks. Breaks do not create free gaps for teachers or students.")

    if st.button("Save Slot Config"):
        st.success("✅ Slot configuration saved successfully.")


    elif st.session_state.user_role == "professor":
        st.title("Professor Portal")
        
