from app.services.pdf_processor import *

pdf_path = 'data/contrato.pdf'
texto_con_coord = extract_text_and_coordinates_by_line(pdf_path)
print(texto_con_coord)
coord_name = find_name_coordinates(pdf_path, 'Lorena Valencia')
print(coord_name)

result = process_pdf(pdf_path, ['Lorena Valencia', 'Armando Colmenares'])
print(result)
