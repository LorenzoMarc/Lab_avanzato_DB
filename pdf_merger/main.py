from pypdf import PdfMerger

pdfs = ['autocertificazione-residenza_Firmata.pdf', 'ID_Marcon_Lorenzo.pdf']

merger = PdfMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write("result.pdf")
merger.close()