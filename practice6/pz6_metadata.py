"""ПЗ6: локальное чтение EXIF, DOCX/PDF и карта GPS/визуальной оценки.

Оригиналы не изменяются. Запуск: python pz6_metadata.py
Пути по умолчанию считаются от папки скрипта, а не текущего каталога.
"""
import argparse
import csv
import html
import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import exifread
import folium
from folium.plugins import MarkerCluster
from PIL import Image, ExifTags
from docx import Document
from pypdf import PdfReader

BASE = Path(__file__).resolve().parent
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
DOC_EXT = {'.docx', '.pdf'}
LOG = logging.getLogger('pz6')


def number(value):
    """Pillow IFDRational, exifread Ratio, пара (числитель, знаменатель)."""
    if hasattr(value, 'num') and hasattr(value, 'den'):
        return value.num / value.den
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def dms_to_decimal(dms, reference):
    """Собственная конвертация: знак * (D + M/60 + S/3600)."""
    if isinstance(reference, bytes):
        reference = reference.decode('ascii')
    reference = str(reference).strip().strip('\x00').upper()
    if reference not in {'N', 'S', 'E', 'W'} or len(dms) != 3:
        raise ValueError('Нужны три DMS-компонента и направление N/S/E/W')
    degrees, minutes, seconds = (number(x) for x in dms)
    if not all(math.isfinite(x) for x in (degrees, minutes, seconds)):
        raise ValueError('Нефинитная координата')
    if degrees < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError('Некорректные компоненты DMS')
    result = degrees + minutes / 60 + seconds / 3600
    limit = 90 if reference in {'N', 'S'} else 180
    if result > limit:
        raise ValueError('Координата за пределами допустимого диапазона')
    return -result if reference in {'S', 'W'} else result


def gps_coordinates(gps):
    keys = ('GPSLatitude', 'GPSLatitudeRef', 'GPSLongitude', 'GPSLongitudeRef')
    if not all(k in gps for k in keys):
        return None
    return (dms_to_decimal(gps[keys[0]], gps[keys[1]]),
            dms_to_decimal(gps[keys[2]], gps[keys[3]]))


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2), encoding='utf-8')


def read_image(path):
    errors, pillow_gps, fields, raw = [], {}, {}, {}
    with Image.open(path) as image:
        raw['size'] = image.size
        raw['format'] = image.format
        exif = image.getexif()
        fields = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        raw['IFD0'] = dict(fields)
        # EXIF/GPS/Interop IFD разворачиваются отдельно, а не остаются смещениями.
        for code, label in ((34665, 'ExifIFD'), (34853, 'GPSIFD'), (40965, 'InteropIFD')):
            try:
                tag_names = ExifTags.GPSTAGS if code == 34853 else ExifTags.TAGS
                nested = {tag_names.get(k, str(k)): v for k, v in exif.get_ifd(code).items()}
                raw[label] = nested
                if code == 34853:
                    pillow_gps = nested
                else:
                    fields.update(nested)
            except Exception as exc:
                errors.append(f'Pillow {label}: {exc}')
    other_gps, tags = {}, {}
    try:
        with path.open('rb') as stream:
            tags = exifread.process_file(stream, details=True)
        for key, tag in tags.items():
            if key.startswith('GPS '):
                name = key[4:]
                other_gps[name] = str(tag) if name.endswith('Ref') else tag.values
    except Exception as exc:
        errors.append(f'exifread: {exc}')
    coordinates = []
    for parser, gps in [('Pillow', pillow_gps), ('exifread', other_gps)]:
        try:
            coordinates.append(gps_coordinates(gps))
        except Exception as exc:
            errors.append(f'{parser} GPS: {exc}')
            coordinates.append(None)
    first, second = coordinates
    if first is not None and second is not None:
        comparison = 'совпадают' if max(abs(a-b) for a, b in zip(first, second)) < 1e-6 else 'РАСХОЖДЕНИЕ'
    elif first is not None or second is not None:
        comparison = 'координаты прочитаны только одним парсером'
    else:
        comparison = 'координаты не извлечены обоими парсерами'
    position = first if first is not None else second
    gps_present = bool(pillow_gps or other_gps or 34853 in exif)
    row = {'filename': path.name, 'camera_make': fields.get('Make', ''),
           'camera_model': fields.get('Model', ''), 'datetime_original': fields.get('DateTimeOriginal', ''),
           'offset_time_original': fields.get('OffsetTimeOriginal', ''),
           'exposure_time': fields.get('ExposureTime', ''), 'f_number': fields.get('FNumber', ''),
           'iso': fields.get('ISOSpeedRatings', fields.get('PhotographicSensitivity', '')),
           'focal_length': fields.get('FocalLength', ''), 'gps_present': gps_present,
           'gps_status': 'valid' if position is not None else ('incomplete' if gps_present else 'absent'),
           'latitude': position[0] if position is not None else '',
           'longitude': position[1] if position is not None else '',
           'coordinate_source': 'Pillow' if first is not None else ('exifread' if second is not None else ''),
           'crosscheck': comparison, 'errors': '; '.join(errors)}
    raw.update(pillow_gps=pillow_gps, exifread={k: str(v) for k, v in tags.items()},
               pillow_coordinates=first, exifread_coordinates=second, summary=row)
    return row, raw


def read_document(path):
    row = dict(filename=path.name, format=path.suffix.lower(), author='', last_modified_by='',
               created='', modified='', revision='', title='', creator='', producer='', errors='')
    if path.suffix.lower() == '.docx':
        properties = Document(path).core_properties
        raw = {key: getattr(properties, key) for key in (
            'author', 'last_modified_by', 'created', 'modified', 'revision', 'title',
            'subject', 'keywords', 'comments', 'category', 'identifier', 'language', 'version')}
        row.update({key: raw[key] for key in row if key in raw})
    else:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(''):
            raise ValueError('PDF защищён паролем; используйте собственную незашифрованную копию')
        info = reader.metadata
        raw = {'Info': dict(info or {})}
        if info:
            for key, attr in [('author', 'author'), ('title', 'title'), ('creator', 'creator'),
                              ('producer', 'producer'), ('created', 'creation_date'), ('modified', 'modification_date')]:
                try:
                    row[key] = getattr(info, attr) or ''
                except Exception as exc:
                    row['errors'] += f'{attr}: {exc}; '
        try:
            xmp = reader.xmp_metadata
            raw['XMP'] = xmp.stream.get_data().decode('utf-8', errors='replace') if xmp else None
        except Exception as exc:
            raw['XMP_error'] = str(exc)
            row['errors'] += f'XMP: {exc}; '
        # В PDF нет стандартных эквивалентов last_modified_by и revision.
    return row, raw


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    if not rows:
        return 'Нет данных.\n'
    def cell(value):
        return str(value if value not in ('', None) else 'не указано').replace('|', '\\|').replace('\n', ' ')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    return '\n'.join(['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |'] +
                     ['| ' + ' | '.join(cell(row.get(k, '')) for k in keys) + ' |' for row in rows]) + '\n'


def visual_point(path, images):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('latitude') is None or data.get('longitude') is None:
        return None, 'Визуальная гипотеза ещё не заполнена'
    matching = next((r for r in images if r['filename'] == data.get('filename')), None)
    if not matching or matching.get('gps_status') != 'absent' or matching.get('errors'):
        return None, 'Для визуальной гипотезы нужен успешно прочитанный снимок без GPS и ошибок'
    lat, lon = float(data['latitude']), float(data['longitude'])
    if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('Неверные координаты визуальной гипотезы')
    if not data.get('place') or not data.get('map_url') or not any(
            o.get('observation') and o.get('inference') for o in data.get('observations', [])):
        return None, 'Заполните место, ссылку на карту и цепочку наблюдение → вывод'
    radius = data.get('accuracy_m')
    if radius is not None and (not math.isfinite(float(radius)) or float(radius) <= 0):
        raise ValueError('accuracy_m должен быть положительным числом или null')
    return dict(filename=data['filename'], source='visual', latitude=lat, longitude=lon,
                place=data['place'], accuracy_m=radius, confidence=data.get('confidence', '')), ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=BASE / 'test_data')
    parser.add_argument('--out', type=Path, default=BASE / 'out_pz6')
    parser.add_argument('--visual', type=Path, default=BASE / 'visual_location.json')
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error('Не найдена папка test_data')
    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / 'raw'
    raw_dir.mkdir(exist_ok=True)
    # Удалённые из набора файлы не должны оставаться в актуальных результатах.
    current_raw_names = {p.name + '.json' for p in args.data.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXT | DOC_EXT}
    stale_raw = [p for p in raw_dir.glob('*.json') if p.name not in current_raw_names]
    if stale_raw:
        archive = args.out / '_previous' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') / 'raw'
        archive.mkdir(parents=True)
        for old_file in stale_raw:
            shutil.move(str(old_file), str(archive / old_file.name))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / 'pz6.log', mode='w', encoding='utf-8')])
    images, documents, problems = [], [], []
    for path in sorted(args.data.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT | DOC_EXT:
            continue
        is_image = path.suffix.lower() in IMAGE_EXT
        try:
            row, raw = read_image(path) if is_image else read_document(path)
            save_json(raw_dir / (path.name + '.json'), raw)
            if row.get('errors'):
                problems.append(f'{path.name}: {row["errors"]}')
        except Exception as exc:
            row = {'filename': path.name, 'errors': str(exc)}
            problems.append(f'{path.name}: {exc}')
            LOG.exception('Ошибка чтения %s', path.name)
        (images if is_image else documents).append(row)
        LOG.info('Обработан %s', path.name)
    journal = args.data / 'journal.csv'
    checks = []
    if journal.exists():
        with journal.open(encoding='utf-8-sig', newline='') as stream:
            entries = list(csv.DictReader(stream))
        actual = {r['filename']: r for r in images + documents}
        for entry in entries:
            row = actual.get(entry['filename'])
            result = 'файл отсутствует' if row is None else 'ok'
            if row is not None and row.get('errors'):
                result = 'ошибка чтения или неполные данные'
            elif row is not None and entry.get('kind') == 'image':
                expected = entry.get('expected_gps')
                wanted = {'yes': 'valid', 'no': 'absent'}.get(expected)
                if wanted is None or row.get('gps_status') != wanted:
                    result = 'GPS не соответствует ожиданию / ожидание не задано'
            if not entry.get('obtained_at') or entry.get('origin') not in {'own', 'synthetic', 'educational'}:
                result += '; заполнить дату и допустимое происхождение'
            checks.append({**entry, 'check': result})
        registered = {e['filename'] for e in entries}
        for name in actual.keys() - registered:
            checks.append({'filename': name, 'check': 'файл не внесён в журнал'})
    else:
        problems.append('Нет test_data/journal.csv')
    if not 4 <= len(images) <= 6:
        problems.append('По заданию требуется 4–6 изображений')
    if not 2 <= len(documents) <= 3:
        problems.append('По заданию требуется 2–3 документа')
    if not any(r.get('gps_status') == 'valid' for r in images) or not any(r.get('gps_status') == 'absent' for r in images):
        problems.append('Нужны снимки и с корректным GPS, и без GPS')
    points = [dict(filename=r['filename'], source='GPS EXIF', latitude=r['latitude'], longitude=r['longitude'])
              for r in images if r.get('gps_status') == 'valid']
    try:
        visual, message = visual_point(args.visual, images)
        if visual:
            points.append(visual)
        if message:
            problems.append(message)
    except Exception as exc:
        problems.append(f'Визуальная гипотеза: {exc}')
    for name, rows in [('images_exif.csv', images), ('documents_metadata.csv', documents),
                       ('journal_check.csv', checks), ('points.csv', points)]:
        write_csv(args.out / name, rows)
    if points:
        map_obj = folium.Map(location=[points[0]['latitude'], points[0]['longitude']], zoom_start=12)
        cluster = MarkerCluster(name='GPS EXIF — синий; visual — оранжевый',
                                options={'spiderfyOnMaxZoom': True}).add_to(map_obj)
        for point in points:
            location = [point['latitude'], point['longitude']]
            is_visual = point['source'] == 'visual'
            text = html.escape(f"{point['filename']} | {point['source']} | {point.get('place', '')}")
            folium.Marker(location, tooltip=text, popup=text,
                          icon=folium.Icon(color='orange' if is_visual else 'blue', icon='info-sign')).add_to(cluster)
            if is_visual and point.get('accuracy_m'):
                folium.Circle(location, radius=float(point['accuracy_m']), color='orange', fill=True).add_to(map_obj)
        if len(points) > 1:
            map_obj.fit_bounds([[p['latitude'], p['longitude']] for p in points], max_zoom=16)
        folium.LayerControl(collapsed=False).add_to(map_obj)
        map_obj.save(str(args.out / 'map.html'))
    elif (args.out / 'map.html').exists():
        (args.out / 'map.html').unlink()  # не оставлять карту от прошлого запуска
    problems.extend(f"{r['filename']}: {r['check']}" for r in checks if r['check'] != 'ok')
    summary = dict(run_at_utc=datetime.now(timezone.utc).isoformat(), image_count=len(images),
                   document_count=len(documents), point_count=len(points), issues=problems,
                   gps_image_count=sum(r.get('gps_status') == 'valid' for r in images),
                   visual_point_count=sum(p['source'] == 'visual' for p in points),
                   unique_coordinate_count=len({(p['latitude'], p['longitude']) for p in points}),
                   status='needs_attention' if problems else 'ready_for_manual_review')
    save_json(args.out / 'summary.json', summary)
    report = '# Таблицы для переноса в отчёт\n\n'
    report += 'Пустые поля означают отсутствие значения, а не нулевое значение.\n\n'
    for title, rows in [('Журнал', checks), ('EXIF', images), ('Документы', documents), ('Координаты', points)]:
        report += '## ' + title + '\n\n' + markdown_table(rows) + '\n'
    (args.out / 'tables_for_report.md').write_text(report, encoding='utf-8')
    for problem in problems:
        LOG.warning(problem)
    LOG.info('Готово: %s. Статус: %s', args.out.resolve(), summary['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
