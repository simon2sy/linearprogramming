import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------------------------
# LP TEXT PARSER
# -------------------------------------------------

def parse_objective(text):
    """Returns (coefficients_list, is_minimization)."""
    upper = text.upper()
    minimize = "MIN" in upper
    text_clean = text.replace(" ", "")
    matches = re.findall(r"([+-]?\d*\.?\d*)x(\d+)", text_clean)
    if not matches:
        raise ValueError("No variables found in objective function.")
    max_idx = max(int(idx) for _, idx in matches)
    result = [0.0] * max_idx
    for coef, idx in matches:
        i = int(idx) - 1
        if coef in ("", "+"):
            result[i] = 1.0
        elif coef == "-":
            result[i] = -1.0
        else:
            result[i] = float(coef)
    return result, minimize


def parse_constraint(line):
    line = line.strip()
    # normalise common typos: <+ treated as <=
    line = line.replace("<+", "<=").replace("+>", ">=")
    if "<=" in line:
        sign = "<="
        left, right = line.split("<=", 1)
    elif ">=" in line:
        sign = ">="
        left, right = line.split(">=", 1)
    elif "=" in line:
        sign = "="
        left, right = line.split("=", 1)
    else:
        raise ValueError(f"Cannot parse constraint: '{line}'. Use <=, >= or =")

    # extract (index, coefficient) pairs based on variable number e.g. x3 -> index 2
    matches = re.findall(r"([+-]?\d*\.?\d*)x(\d+)", left.replace(" ", ""))
    if not matches:
        raise ValueError(f"No variables found in constraint: '{line}'")

    max_idx = max(int(idx) for _, idx in matches)
    row = [0.0] * max_idx
    for coef, idx in matches:
        i = int(idx) - 1
        if coef in ("", "+"):
            row[i] = 1.0
        elif coef == "-":
            row[i] = -1.0
        else:
            row[i] = float(coef)

    return row, float(right.strip()), sign


# helper: format a numeric value — show M symbol if it equals ±M
def fmt(val, M=1_000_000):
    v = float(val)
    if abs(v - (-M))   < 1: return "-M"
    if abs(v -   M)    < 1: return  "M"
    if abs(v - (-2*M)) < 1: return "-2M"
    if abs(v -  (2*M)) < 1: return  "2M"
    if abs(v - (-3*M)) < 1: return "-3M"
    if abs(v -  (3*M)) < 1: return  "3M"
    return str(round(v, 4))


class SimplexSolver:

    M = 1_000_000

    def __init__(self, objective, A, b, signs, minimize=False):
        self.minimize = minimize
        obj = np.array(objective, dtype=float)
        self.obj = -obj if minimize else obj  # Min->Max conversion
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.signs = signs
        self.rows, self.cols = self.A.shape
        self._build_standard_form()

    def _build_standard_form(self):
        self.var_names = [f"x{i+1}" for i in range(self.cols)]
        cj       = list(self.obj)   # decision variable costs
        extra_cols = []             # extra columns in the ORDER they are added
        self.bv  = []
        self.cb  = []

        for i in range(self.rows):
            if self.signs[i] == "<=":
                s = np.zeros((self.rows, 1)); s[i, 0] = 1
                extra_cols.append(s)
                self.var_names.append(f"s{i+1}")
                cj.append(0.0)
                self.bv.append(f"s{i+1}")
                self.cb.append(0.0)

            elif self.signs[i] == ">=":
                # surplus
                s = np.zeros((self.rows, 1)); s[i, 0] = -1
                extra_cols.append(s)
                self.var_names.append(f"s{i+1}")
                cj.append(0.0)
                # artificial
                a = np.zeros((self.rows, 1)); a[i, 0] = 1
                extra_cols.append(a)
                self.var_names.append(f"a{i+1}")
                cj.append(-self.M)
                self.bv.append(f"a{i+1}")
                self.cb.append(-self.M)

            else:  # "="
                a = np.zeros((self.rows, 1)); a[i, 0] = 1
                extra_cols.append(a)
                self.var_names.append(f"a{i+1}")
                cj.append(-self.M)
                self.bv.append(f"a{i+1}")
                self.cb.append(-self.M)

        if extra_cols:
            self.A = np.hstack([self.A] + extra_cols)

        self.cj = np.array(cj, dtype=float)
        self.cb = np.array(self.cb, dtype=float)

    def _compute_zj(self):
        zj = self.cb @ self.A          # shape (n_cols,)
        zj_b = float(self.cb @ self.b) # objective value
        return zj, zj - self.cj, zj_b

    def solve(self):
        tables = []   # each entry: (bv, cb, A, b, zj, zj_cj, zj_b, pivot_info)
        for _ in range(500):  # iteration cap
            zj, zj_cj, zj_b = self._compute_zj()

            # Optimality: all Zj-Cj >= 0 (same rule, Min converted to Max)
            if np.all(zj_cj >= -1e-9):
                tables.append((
                    self.bv.copy(), self.cb.copy(),
                    self.A.copy(), self.b.copy(),
                    zj.copy(), zj_cj.copy(), zj_b, None
                ))
                break

            # Pivot column: most negative Zj-Cj
            pivot_col = int(np.argmin(zj_cj))

            ratios = np.where(
                self.A[:, pivot_col] > 1e-9,
                self.b / self.A[:, pivot_col],
                np.inf
            )
            pivot_row = int(np.argmin(ratios))

            if ratios[pivot_row] == np.inf:
                raise ValueError("Unbounded solution — problem has no finite optimum.")

            entering = self.var_names[pivot_col]
            leaving  = self.bv[pivot_row]
            piv_val  = float(self.A[pivot_row, pivot_col])
            ratio_list = [round(float(r), 4) if r != np.inf else "—" for r in ratios]

            pivot_info = {
                "entering":   entering,
                "leaving":    leaving,
                "pivot_row":  pivot_row,
                "pivot_col":  pivot_col,
                "pivot_val":  round(piv_val, 4),
                "ratios":     ratio_list,
                "old_pivot_row": self.A[pivot_row].copy(),
                "old_b_pivot":  float(self.b[pivot_row]),
                "minimize":   self.minimize,
            }

            tables.append((
                self.bv.copy(), self.cb.copy(),
                self.A.copy(), self.b.copy(),
                zj.copy(), zj_cj.copy(), zj_b, pivot_info
            ))

            # Pivot
            self.A[pivot_row] /= piv_val
            self.b[pivot_row] /= piv_val

            # Store new row operations for display
            row_ops = []
            for i in range(self.rows):
                if i != self.rows:  # always true, kept for clarity
                    if i != pivot_row:
                        f = float(self.A[i, pivot_col])
                        if abs(f) > 1e-12:
                            row_ops.append({
                                "row_idx": i,
                                "row_var": self.bv[i],
                                "factor":  round(f, 4),
                            })
                        self.A[i] -= f * self.A[pivot_row]
                        self.b[i] -= f * self.b[pivot_row]

            pivot_info["row_ops"]      = row_ops
            pivot_info["new_pivot_row"] = self.A[pivot_row].copy()
            pivot_info["new_b_pivot"]   = float(self.b[pivot_row])

            self.cb[pivot_row] = self.cj[pivot_col]
            self.bv[pivot_row] = self.var_names[pivot_col]

        return tables


# -------------------------------------------------
# GUI
# -------------------------------------------------

class LPApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Linear Programming Solver (Simplex)")
        self.root.geometry("1100x750")
        self.tables = []
        self.var_names = []
        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(pady=10)

        tk.Label(top, text="Objective Function  (e.g.  Max Z = 3x1 + 5x2)").pack()
        self.obj_entry = tk.Entry(top, width=70)
        self.obj_entry.pack()

        tk.Label(top, text="Constraints — one per line  (e.g.  2x1 + 3x2 <= 8)").pack(pady=5)
        self.constraints = tk.Text(top, height=8, width=70)
        self.constraints.pack()

        btn = tk.Frame(self.root)
        btn.pack(pady=8)
        tk.Button(btn, text="Solve",      command=self._solve,      width=12).pack(side=tk.LEFT, padx=8)
        tk.Button(btn, text="Export PDF", command=self._export_pdf, width=12).pack(side=tk.LEFT)

        self.output = tk.Text(self.root, width=130, height=32, font=("Courier", 9))
        scroll = tk.Scrollbar(self.root, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _print_table(self, table, iteration):
        bv, cb, A, b, zj, zj_cj, zj_b, pivot_info = table
        w = 9  # column width

        self.output.insert(tk.END, f"\n{'='*60}\n Iteration {iteration}\n{'='*60}\n")

        # Header
        header = f"{'BV':<8}{'CB':>{w}}"
        for v in self.var_names:
            header += f"{v:>{w}}"
        header += f"{'RHS':>{w}}  {'Ratio':>{w}}\n"
        self.output.insert(tk.END, header)

        # Rows — mark pivot row with >>>
        for i in range(len(bv)):
            marker = ">>>" if (pivot_info and i == pivot_info["pivot_row"]) else "   "
            row = f"{marker}{bv[i]:<5}{fmt(cb[i]):>{w}}"
            for val in A[i]:
                row += f"{fmt(val):>{w}}"
            row += f"{fmt(b[i]):>{w}}"
            # ratio column
            if pivot_info:
                r = pivot_info["ratios"][i]
                row += f"  {str(r):>{w}}"
            row += "\n"
            self.output.insert(tk.END, row)

        # Zj
        zj_row = f"   {'Zj':<5}{' ':>{w}}"
        for val in zj:
            zj_row += f"{fmt(val):>{w}}"
        zj_row += f"{fmt(zj_b):>{w}}\n"
        self.output.insert(tk.END, zj_row)

        # Zj-Cj
        zjcj_row = f"   {'Zj-Cj':<5}{' ':>{w}}"
        for val in zj_cj:
            zjcj_row += f"{fmt(val):>{w}}"
        zjcj_row += "\n"
        self.output.insert(tk.END, zjcj_row)

        # Pivot info block
        if pivot_info:
            self._print_pivot_info(pivot_info)

    def _print_pivot_info(self, p):
        entering  = p["entering"]
        leaving   = p["leaving"]
        piv_val   = p["pivot_val"]
        piv_row   = p["pivot_row"]
        piv_col   = p["pivot_col"]
        old_row   = p["old_pivot_row"]
        old_b     = p["old_b_pivot"]
        new_row   = p["new_pivot_row"]
        new_b     = p["new_b_pivot"]
        row_ops   = p["row_ops"]

        self.output.insert(tk.END, f"\n  >> PIVOT OPERATION\n")
        rule = "most negative Zj-Cj"
        self.output.insert(tk.END, f"     Entering variable : {entering}  ({rule} → pivot column {piv_col+1})\n")
        self.output.insert(tk.END, f"     Leaving  variable : {leaving}   (minimum ratio → pivot row {piv_row+1})\n")
        self.output.insert(tk.END, f"     Pivot element     : {fmt(piv_val)}\n")
        self.output.insert(tk.END, f"     {leaving} is replaced by {entering} in the basis.\n")

        # New pivot row
        old_row_str = "  ".join(fmt(v) for v in old_row)
        new_row_str = "  ".join(fmt(v) for v in new_row)
        self.output.insert(tk.END, f"\n  New pivot row  (R{piv_row+1} = R{piv_row+1} / {fmt(piv_val)}):\n")
        self.output.insert(tk.END, f"     Old R{piv_row+1}: [{old_row_str}  | {fmt(old_b)}]\n")
        self.output.insert(tk.END, f"     New R{piv_row+1}: [{new_row_str}  | {fmt(new_b)}]\n")

        # Other row updates
        if row_ops:
            self.output.insert(tk.END, f"\n  Other row updates:\n")
            for op in row_ops:
                ri   = op["row_idx"]
                rvar = op["row_var"]
                f    = op["factor"]
                sign = "-" if f > 0 else "+"
                self.output.insert(tk.END,
                    f"     R{ri+1} ({rvar}) = R{ri+1} {sign} ({abs(f)}) x R{piv_row+1}\n")

        self.output.insert(tk.END, "\n" + "-"*60 + "\n")

    def _solve(self):
        try:
            self.output.delete("1.0", tk.END)

            obj_text = self.obj_entry.get().strip()
            if not obj_text:
                raise ValueError("Please enter an objective function.")

            objective, minimize = parse_objective(obj_text)

            raw_lines = self.constraints.get("1.0", tk.END).strip().split("\n")
            lines = [l.strip() for l in raw_lines if l.strip()]
            if not lines:
                raise ValueError("Please enter at least one constraint.")

            A, b, signs = [], [], []
            for line in lines:
                row, rhs, sign = parse_constraint(line)
                A.append(row)
                b.append(rhs)
                signs.append(sign)

            # pad all rows to the same length as the objective
            n_vars = len(objective)
            for row in A:
                while len(row) < n_vars:
                    row.append(0.0)
                if len(row) > n_vars:
                    raise ValueError(
                        f"A constraint has more variables than the objective function. "
                        f"Check your input (objective has {n_vars} variables).")
            solver = SimplexSolver(objective, A, b, signs, minimize)
            self.tables = solver.solve()
            self.var_names = solver.var_names

            for i, t in enumerate(self.tables):
                self._print_table(t, i + 1)

            self._print_optimal(minimize, solver.cols)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _print_optimal(self, minimize, n_decision_vars):
        last = self.tables[-1]
        bv_final, _, _, b_final, _, _, zj_b, _ = last

        # map every basic variable to its value
        bv_vals = {name: round(float(val), 6) for name, val in zip(bv_final, b_final)}

        # all decision variables — non-basic ones are 0
        dec_vars = [f"x{i+1}" for i in range(n_decision_vars)]
        var_vals = {v: bv_vals.get(v, 0.0) for v in dec_vars}

        label   = "Min Z" if minimize else "Max Z"
        opt_val = round(-float(zj_b) if minimize else float(zj_b), 6)

        sep  = "*" * 50
        sep2 = "-" * 50

        self.output.insert(tk.END, f"\n\n{sep}\n")
        self.output.insert(tk.END,  "*          OPTIMAL SOLUTION FOUND             *\n")
        self.output.insert(tk.END, f"{sep}\n\n")

        self.output.insert(tk.END, "  Decision Variable Values:\n")
        self.output.insert(tk.END, f"  {sep2}\n")
        for v in dec_vars:
            val = var_vals[v]
            self.output.insert(tk.END, f"  {v}  =  {val}\n")

        self.output.insert(tk.END, f"\n  {sep2}\n")
        self.output.insert(tk.END, f"  Optimal Value:  {label}  =  {opt_val}\n")
        self.output.insert(tk.END, f"  {sep2}\n")

        # one-line summary  e.g.  x1 = 2.0,  x2 = 3.0,  Max Z = 16.0
        summary = ",   ".join(f"{v} = {var_vals[v]}" for v in dec_vars)
        summary += f",   {label} = {opt_val}"
        self.output.insert(tk.END, f"\n  Summary:  {summary}\n")
        self.output.insert(tk.END, f"\n{sep}\n")

        self.output.see(tk.END)

    def _export_pdf(self):
        if not self.tables:
            messagebox.showerror("Error", "Solve a problem first.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not path:
            return

        styles = getSampleStyleSheet()
        elements = []

        for i, table in enumerate(self.tables):
            bv, cb, A, b, zj, zj_cj, zj_b, pivot_info = table
            elements.append(Paragraph(f"Iteration {i+1}", styles["Heading3"]))

            header = ["BV", "CB"] + self.var_names + ["RHS"]
            data = [header]

            for r in range(len(bv)):
                row = [str(bv[r]), fmt(cb[r])]
                row += [fmt(v) for v in A[r]]
                row.append(fmt(b[r]))
                data.append(row)

            zj_row   = ["Zj",    ""] + [fmt(v) for v in zj]    + [fmt(zj_b)]
            zjcj_row = ["Zj-Cj", ""] + [fmt(v) for v in zj_cj] + [""]
            data.append(zj_row)
            data.append(zjcj_row)

            elements.append(Table(data, repeatRows=1))
            elements.append(Spacer(1, 8))

            # Pivot info in PDF
            if pivot_info:
                p = pivot_info
                elements.append(Paragraph(
                    f"Pivot: {p['leaving']} leaves, {p['entering']} enters | "
                    f"Pivot element = {p['pivot_val']} | "
                    f"New R{p['pivot_row']+1} = R{p['pivot_row']+1} / {p['pivot_val']}",
                    styles["Normal"]
                ))
                for op in p.get("row_ops", []):
                    ri = op["row_idx"]; f = op["factor"]; sign = "-" if f > 0 else "+"
                    elements.append(Paragraph(
                        f"R{ri+1} ({op['row_var']}) = R{ri+1} {sign} ({abs(f)}) x R{p['pivot_row']+1}",
                        styles["Normal"]
                    ))
            elements.append(Spacer(1, 14))

        # Optimal solution section in PDF
        last = self.tables[-1]
        bv_final, _, _, b_final, _, _, zj_b, _ = last
        bv_vals  = {name: round(float(val), 6) for name, val in zip(bv_final, b_final)}
        minimize = "Min" in self.obj_entry.get().upper()
        n_dec    = sum(1 for v in self.var_names if v.startswith("x"))
        dec_vars = [f"x{i+1}" for i in range(n_dec)]
        var_vals = {v: bv_vals.get(v, 0.0) for v in dec_vars}
        label    = "Min Z" if minimize else "Max Z"
        opt_val  = round(-float(zj_b) if minimize else float(zj_b), 6)

        elements.append(Paragraph("Optimal Solution", styles["Heading2"]))
        elements.append(Paragraph("Decision Variable Values:", styles["Normal"]))
        for v in dec_vars:
            elements.append(Paragraph(f"  {v}  =  {var_vals[v]}", styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"Optimal Value:  {label}  =  {opt_val}", styles["Normal"]))
        summary = ",   ".join(f"{v} = {var_vals[v]}" for v in dec_vars) + f",   {label} = {opt_val}"
        elements.append(Paragraph(f"Summary:  {summary}", styles["Normal"]))

        SimpleDocTemplate(path).build(elements)
        messagebox.showinfo("Saved", "PDF exported successfully.")


# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    LPApp(root)
    root.mainloop()
