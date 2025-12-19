import tkinter as tk

root = tk.Tk()
root.title("Test Window")
root.geometry("300x200")

label = tk.Label(root, text="If you see this, tkinter works!", font=('Arial', 14))
label.pack(pady=20)

button = tk.Button(root, text="Click Me", command=lambda: print("Button works!"))
button.pack()

root.mainloop()