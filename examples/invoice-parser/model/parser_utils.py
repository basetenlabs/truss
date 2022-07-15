
import requests
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

# These values must be provided for this service to run.
PROJECT_ID = 'sample'
LOCATION = 'sample'
PROCESSOR_ID = 'sample'
CONFIDENCE_THRESHOLD = 0.5


def extract_pdf_data(file_url: str, creds: dict):
    '''
    Extracts structured data from pdfs.
    Args:
        file_url: The url for a pdf file.
    '''

    credentials = service_account.Credentials.from_service_account_info(creds)
    client = documentai.DocumentProcessorServiceClient(credentials=credentials)
    res = requests.get(file_url)
    input_config = documentai.types.RawDocument(
        content=res.content, mime_type='application/pdf')
    name = f'projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}'

    request = {"name": name, "raw_document": input_config}
    result = client.process_document(request=request)
    document = result.document

    extracted_data = extract_invoice_data(document)

    return extracted_data


def extract_invoice_data(document):
    extracted_data = []
    document_lines = []
    for page in document.pages:
        document_lines.extend(page.lines)

    for entity in document.entities:
        bounding_boxes = get_line_bbox_for_segment(
            entity.text_anchor.text_segments[0].start_index,
            entity.text_anchor.text_segments[-1].end_index, document_lines)
        extract = {
            'field': entity.type_,
            'field_confidence': entity.confidence,
            'field_bbox': bounding_boxes,
            'field_value': entity.mention_text,
            'field_value_confidence': entity.confidence,
            'field_value_bbox': bounding_boxes,
        }
        if extract['field_confidence'] > CONFIDENCE_THRESHOLD and\
                extract['field_value_confidence'] > CONFIDENCE_THRESHOLD:
            extracted_data.append(extract)
    return extracted_data


def get_line_bbox_for_segment(segment_min, segment_max, lines):
    matching_lines = [
        li for li in lines
        if li.layout.text_anchor.text_segments[0].start_index <= segment_min
        and segment_max <= li.layout.text_anchor.text_segments[-1].end_index
    ]
    matching_lines += [
        li for li in lines
        if li.layout.text_anchor.text_segments[0].start_index >= segment_min
        and li.layout.text_anchor.text_segments[-1].end_index <= segment_max
    ]
    x_min = min([
        vertex.x for vertex in min(matching_lines,
                                   key=lambda x: [
                                       vertex.x for vertex in x.layout.
                                       bounding_poly.normalized_vertices
                                   ]).layout.bounding_poly.normalized_vertices
    ])
    x_max = max([
        vertex.x for vertex in max(matching_lines,
                                   key=lambda x: [
                                       vertex.x for vertex in x.layout.
                                       bounding_poly.normalized_vertices
                                   ]).layout.bounding_poly.normalized_vertices
    ])
    y_min = min([
        vertex.y for vertex in min(matching_lines,
                                   key=lambda x: [
                                       vertex.y for vertex in x.layout.
                                       bounding_poly.normalized_vertices
                                   ]).layout.bounding_poly.normalized_vertices
    ])
    y_max = max([
        vertex.y for vertex in max(matching_lines,
                                   key=lambda x: [
                                       vertex.y for vertex in x.layout.
                                       bounding_poly.normalized_vertices
                                   ]).layout.bounding_poly.normalized_vertices
    ])
    return [{
        'x': x_min,
        'y': y_min
    }, {
        'x': x_max,
        'y': y_min
    }, {
        'x': x_max,
        'y': y_max
    }, {
        'x': x_min,
        'y': y_max
    }]


def dict_from_vertices(vertices):
    return [{'x': v.x, 'y': v.y} for v in vertices]
