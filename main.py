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

# Simple user database
USER_DB = {
    "uv": {"password": "123", "role": "university"},
    "st": {"password": "321", "role": "student"},
    "pro": {"password": "111", "role": "professor"},
}

def login():
    st.title("Login")
    username = st.text_input("User  ID")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_DB and USER_DB[username]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = USER_DB[username]["role"]
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def logout():
    if st.button("Logout"):
        for key in ["logged_in", "username", "role"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
else:
    role = st.session_state["role"]
    st.sidebar.write(f"Logged in as: {st.session_state['username']} ({role})")
    logout()

    if role == "university":
        # Paste your entire existing UI code here (from st.set_page_config to the end)
        # For brevity, I will just call a function that you should define with your UI code.
        def university_ui():
            st.set_page_config(page_title="AI Timetable Generator", layout="wide")
            st.title("Time Table Generator")

            with st.expander("About this app", expanded=True):
                st.write("Upload your data files (CSV/Excel). Columns expected are listed below each uploader. Use semicolon (;) separated lists for multi-values (e.g., can_teach: ENG101;MAT101).")

            col1, col2 = st.columns([2,1])

            with col1:
                st.subheader("Upload data files")
                courses_file = st.file_uploader("Courses (CSV/XLSX)", type=["csv","xlsx"], key="courses")
                sections_file = st.file_uploader("Sections (CSV/XLSX)", type=["csv","xlsx"], key="sections")
                faculties_file = st.file_uploader("Faculties (CSV/XLSX)", type=["csv","xlsx"], key="faculties")
                rooms_file = st.file_uploader("Rooms (CSV/XLSX)", type=["csv","xlsx"], key="rooms")
                fixed_file = st.file_uploader("Fixed events (optional)", type=["csv","xlsx"], key="fixed")

            with col2:
                st.subheader("Slot configuration")
                days_text = st.text_input("Days (comma separated)", value="Mon,Tue,Wed,Thu,Fri")
                slots_per_day = st.number_input("Slots per day", min_value=3, max_value=10, value=6)
                time_limit = st.number_input("Solver time limit (seconds)", min_value=5, max_value=300, value=30)

            if courses_file and sections_file and faculties_file and rooms_file:
                def load_df(f):
                    if f.name.endswith('.csv'):
                        return pd.read_csv(f)
                    else:
                        return pd.read_excel(f)

                courses_df = load_df(courses_file)
                sections_df = load_df(sections_file)
                faculties_df = load_df(faculties_file)
                rooms_df = load_df(rooms_file)
                fixed_df = load_df(fixed_file) if fixed_file else None

                st.success("Files loaded — preview below")
                with st.expander("Preview data frames (courses) "):
                    st.dataframe(courses_df)
                with st.expander("Preview (sections)"):
                    st.dataframe(sections_df)
                with st.expander("Preview (faculties)"):
                    st.dataframe(faculties_df)
                with st.expander("Preview (rooms)"):
                    st.dataframe(rooms_df)
                if fixed_df is not None:
                    with st.expander("Preview (fixed events)"):
                        st.dataframe(fixed_df)

                days = [d.strip() for d in days_text.split(",") if d.strip()]
                slots = [(d, i) for d in days for i in range(1, int(slots_per_day)+1)]

                if st.button("Generate Timetable"):
                    with st.spinner("Solving... this may take a while depending on problem size"):
                        try:
                            result_df, status = solve_timetable(
                                courses_df=courses_df,
                                sections_df=sections_df,
                                faculties_df=faculties_df,
                                rooms_df=rooms_df,
                                slots=slots,
                                fixed_events_df=fixed_df,
                                time_limit_sec=int(time_limit),
                            )
                            st.write("Solver status:", status)
                            if result_df.empty:
                                st.warning("No assignments found — check input data, availabilities, and hours_per_week values.")
                            else:
                                st.success("Timetable generated")
                                st.dataframe(result_df)

                                # Download as Excel
                                towrite = io.BytesIO()
                                with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                                    result_df.to_excel(writer, index=False, sheet_name='timetable')
                                towrite.seek(0)
                                st.download_button("Download timetable (Excel)", data=towrite, file_name="timetable.xlsx")

                                # Download as CSV
                                csv = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button("Download timetable (CSV)", data=csv, file_name="timetable.csv")

                                # Simple PDF export using reportlab
                                try:
                                    from reportlab.lib.pagesizes import A4
                                    from reportlab.pdfgen import canvas
                                    pdf_bytes = io.BytesIO()
                                    c = canvas.Canvas(pdf_bytes, pagesize=A4)
                                    text = c.beginText(40, 800)
                                    text.setFont("Helvetica", 10)
                                    for i, row in result_df.iterrows():
                                        line = f"{row['cohort']} | {row['day']}-{row['slot']} | {row['course_id']} ({row['course_name']}) | {row['faculty_id']} | {row['room_id']}"
                                        text.textLine(line)
                                        if text.getY() < 40:
                                            c.drawText(text)
                                            c.showPage()
                                            text = c.beginText(40, 800)
                                            text.setFont("Helvetica", 10)
                                    c.drawText(text)
                                    c.save()
                                    pdf_bytes.seek(0)
                                    st.download_button("Download timetable (PDF)", data=pdf_bytes, file_name="timetable.pdf")
                                except Exception:
                                    st.info("PDF export not available (install reportlab).")

                        except Exception as e:
                            st.error(f"Error while solving: {e}")

            else:
                st.info("Please upload Courses, Sections, Faculties, and Rooms files to continue.")

            st.markdown("---")
            st.caption("Prototype: tweak constraints, soft weights and model parameters in code for better university-specific behaviour.")

        university_ui()

    elif role == "student":
        st.title("Student Portal")
        st.write("Welcome to the student portal.")

    elif role == "professor":
        st.title("Professor Portal")
        st.write("Welcome to the professor portal.")

