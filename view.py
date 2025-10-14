import pandas as pd
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Roster, Attendance
import datetime
import io
from rapidfuzz import process, fuzz
from django.db.models import Q

def to_safe_time(t):
    if pd.isna(t) or t is None: return None
    if isinstance(t, datetime.time): return t
    if isinstance(t, (float, int)):
        total_seconds = int(t * 24 * 60 * 60)
        total_seconds = min(total_seconds, 86399)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return datetime.time(hours, minutes, seconds)
    try:
        return pd.to_datetime(str(t), errors='coerce').time()
    except (ValueError, TypeError):
        return None

class FileUploadView(APIView):
    def post(self, request, *args, **kwargs):
        roster_file = request.FILES.get('roster')
        attendance_file = request.FILES.get('attendance')

        if not roster_file or not attendance_file:
            return Response(
                {'error': 'Please upload both "roster" and "attendance" files.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            roster_df = pd.read_excel(io.BytesIO(roster_file.read()), header=0, skiprows=[1])
            roster_df.columns = [str(c).strip().lower() for c in roster_df.columns]
            attendance_df = pd.read_excel(io.BytesIO(attendance_file.read()), header=None, skiprows=1)
            attendance_df.columns = [
                'ads_id', 'name_in_attendance', 'user_type', 'designation', 'department',
                'location', 'first_in', 'last_out', 'gross_time', 'out_of_office_time',
                'out_of_office_count', 'net_office_time', 'attendance_date'
            ]

            dates = pd.to_datetime(attendance_df['attendance_date'], dayfirst=True, errors='coerce')
            first_valid_date = dates.dropna().iloc[0]
            processing_year, processing_month = first_valid_date.year, first_valid_date.month

            roster_names = roster_df['name'].dropna().unique().tolist()
            attendance_names = attendance_df['name_in_attendance'].dropna().unique().tolist()

            name_map = {}
            for att_name in attendance_names:
                match = process.extractOne(att_name, roster_names, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 90:
                    canonical_name = match[0]
                    name_map[att_name] = canonical_name

            attendance_df['name'] = attendance_df['name_in_attendance'].map(name_map)
            matched_names = list(name_map.values())
            roster_df = roster_df[roster_df['name'].isin(matched_names)]
            attendance_df = attendance_df.dropna(subset=['name'])

            self._process_roster_data(roster_df, processing_year, processing_month)
            self._process_attendance_data(attendance_df)
        except Exception as e:
            return Response({'error': f'An error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {'message': f'Files processed successfully for {first_valid_date.strftime("%B %Y")}. Only matched records were saved.'},
            status=status.HTTP_201_CREATED
        )

    def _process_roster_data(self, df, year, month):
        for _, row in df.iterrows():
            name = row.get('name')
            team = row.get('sr no')
            for col_name, schedule in row.items():
                if str(col_name).isdigit():
                    try:
                        date = datetime.date(year, month, int(col_name))
                        Roster.objects.update_or_create(
                            name=name, date=date,
                            defaults={'team': team, 'schedule': schedule}
                        )
                    except ValueError:
                        continue

    def _process_attendance_data(self, df):
        df.replace({np.nan: None}, inplace=True)
        for _, row in df.iterrows():
            attendance_date = pd.to_datetime(row['attendance_date'], dayfirst=True, errors='coerce').date()
            if not attendance_date: continue

            defaults = {
                'employee_id': row['ads_id'], 'user_type': row['user_type'],
                'designation': row['designation'], 'department': row['department'],
                'location': row['location'], 'first_in': to_safe_time(row['first_in']),
                'last_out': to_safe_time(row['last_out']), 'gross_time': to_safe_time(row['gross_time']),
                'out_of_office_time': to_safe_time(row['out_of_office_time']),
                'out_of_office_count': row['out_of_office_count'],
                'net_office_time': to_safe_time(row['net_office_time'])
            }
            Attendance.objects.update_or_create(
                name=row['name'], date=attendance_date, defaults=defaults
            )

class SearchView(APIView):
    def get(self, request, *args, **kwargs):
        action = request.query_params.get('action')
        if action == 'count':
            return self._perform_count(request)
        elif action == 'low_hours':
            return self._find_low_hours(request)
        elif action == 'non_pl_low_hours':
            return self._find_non_pl_low_hours(request)
        else:
            return self._perform_search(request)

    def _get_date_range(self, request):
        start_date_str = request.query_params.get('start_date')
        end_date_str = request.query_params.get('end_date')
        if start_date_str:
            start_date = pd.to_datetime(start_date_str).date()
            end_date = pd.to_datetime(end_date_str).date() if end_date_str else start_date
        else:
            last_attendance = Attendance.objects.order_by('-date').first()
            if not last_attendance: return None, None
            target_date = last_attendance.date
            start_date = target_date.replace(day=1)
            next_month = start_date.replace(day=28) + datetime.timedelta(days=4)
            end_date = next_month - datetime.timedelta(days=next_month.day)
        return start_date, end_date

    def _perform_count(self, request):
        query = request.query_params.get('q')
        employee_id = request.query_params.get('id')

        if not query and not employee_id:
            return Response({'error': 'A name query (q=...) or an ID query (id=...) is required.'}, status=status.HTTP_400_BAD_REQUEST)

        start_date, end_date = self._get_date_range(request)
        if not start_date:
            return Response({'error': 'No data found for the requested period.'}, status=status.HTTP_404_NOT_FOUND)

        rosters_query = Roster.objects.filter(date__range=[start_date, end_date])
        employee_names = []

        if employee_id:
            employee_names = Attendance.objects.filter(
                employee_id__iexact=employee_id,
                date__range=[start_date, end_date]
            ).values_list('name', flat=True).distinct()
            if not employee_names.exists():
                return Response({'error': f'No employee found with ID "{employee_id}"'}, status=status.HTTP_404_NOT_FOUND)

        elif query:
            temp_rosters = rosters_query
            query_words = query.split()
            for word in query_words:
                temp_rosters = temp_rosters.filter(name__icontains=word)

            employee_names = temp_rosters.values_list('name', flat=True).distinct()
            if not employee_names.exists():
                 return Response({'error': f'No employee found matching "{query}"'}, status=status.HTTP_404_NOT_FOUND)

        results = []
        WFO_SCHEDULES = {'WFO-M', 'WFO-G', 'WFO-G2', 'WFO-S', 'WFO-N'}
        WFH_SCHEDULES = {'WFH-M', 'WFH-G', 'WFH-G2', 'WFH-S', 'WFH-N'}

        for name in employee_names:
            counts = {
                'Total WFO': 0, 'Total WFH': 0, 'Total WO': 0, 'Total PL': 0,
            }
            # Filter the main roster query for the specific name
            employee_rosters = rosters_query.filter(name=name)

            for entry in employee_rosters:
                schedule = str(entry.schedule).upper() if entry.schedule else ""
                if schedule in WFO_SCHEDULES:
                    counts['Total WFO'] += 1
                elif schedule in WFH_SCHEDULES:
                    counts['Total WFH'] += 1
                elif schedule == 'WO':
                    counts['Total WO'] += 1
                elif schedule == 'PL':
                    counts['Total PL'] += 1

            counts['Total working days'] = counts['Total WFO'] + counts['Total WFH']
            counts['Total Leaves'] = counts['Total WO'] + counts['Total PL']

            # Fetch the employee ID for the response
            emp_id = Attendance.objects.filter(name=name).values_list('employee_id', flat=True).first()

            results.append({
                'employee': name,
                'employee_id': emp_id,
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'counts': counts
            })
        return Response(results)

    def _find_low_hours(self, request):
        start_date, end_date = self._get_date_range(request)
        if not start_date:
            return Response({'error': 'No data found.'}, status=404)
        eight_hours = datetime.time(8, 0)
        low_hour_attendances = Attendance.objects.filter(
            date__range=[start_date, end_date],
            net_office_time__lt=eight_hours
        ).order_by('date', 'name')

        roster_map = {
            (r.name, r.date): r for r in Roster.objects.filter(
                date__range=[start_date, end_date],
                name__in=low_hour_attendances.values_list('name', flat=True)
            ).exclude(Q(schedule__iexact='PL') | Q(schedule__iexact='WO'))
        }

        results = []
        for att in low_hour_attendances:
            roster_record = roster_map.get((att.name, att.date))
            if roster_record:
                results.append({
                    'name': att.name,
                    'employee_id': att.employee_id,
                    'team': roster_record.team,
                    'date': att.date,
                    'shift': roster_record.schedule,
                    'net_office_time': att.net_office_time
                })
        return Response({'count': len(results), 'employees_with_low_hours': results})

    def _find_non_pl_low_hours(self, request):
        start_date, end_date = self._get_date_range(request)
        if not start_date: return Response({'error': 'No data found.'}, status=404)
        eight_hours = datetime.time(8, 0)
        low_hour_attendances = Attendance.objects.filter(
            date__range=[start_date, end_date], net_office_time__lt=eight_hours
        ).order_by('date', 'name')
        roster_map = {
            (r.name, r.date): r for r in Roster.objects.filter(
                date__range=[start_date, end_date],
                name__in=low_hour_attendances.values_list('name', flat=True)
            ).exclude(schedule__iexact='PL')
        }
        results = []
        for att in low_hour_attendances:
            roster_record = roster_map.get((att.name, att.date))
            if roster_record:
                results.append({
                    'name': att.name,
                    'employee_id': att.employee_id,
                    'team': roster_record.team, 'date': att.date,
                    'shift': roster_record.schedule, 'net_office_time': att.net_office_time
                })
        return Response({'count': len(results), 'employees': results})

    def _perform_search(self, request):
        teamname = request.query_params.get('teamname')
        query = request.query_params.get('q')
        shift = request.query_params.get('shift')
        employee_id = request.query_params.get('id')
        start_date, end_date = self._get_date_range(request)

        if not start_date:
            return Response({'error': 'No data found for the requested period.'}, status=404)

        rosters = Roster.objects.filter(date__range=[start_date, end_date])

        if employee_id:
            names_with_id = Attendance.objects.filter(
                employee_id__iexact=employee_id,
                date__range=[start_date, end_date]
            ).values_list('name', flat=True).distinct()

            if not names_with_id.exists():
                return Response([])

            rosters = rosters.filter(name__in=names_with_id)

        if teamname:
            rosters = rosters.filter(team__iexact=teamname)

        if query:
            query_words = query.split()
            for word in query_words:
                rosters = rosters.filter(name__icontains=word)

        if shift:
            rosters = rosters.filter(schedule__iexact=shift)

        rosters = rosters.order_by('date', 'name')

        names_to_fetch = rosters.values_list('name', flat=True).distinct()
        attendance_data = Attendance.objects.filter(
            name__in=names_to_fetch,
            date__range=[start_date, end_date]
        )

        attendance_map = {(att.name, att.date): att for att in attendance_data}

        results = []
        for roster in rosters:
            attendance_record = attendance_map.get((roster.name, roster.date))
            results.append({
                'name': roster.name,
                'employee_id': attendance_record.employee_id if attendance_record else None,
                'team': roster.team,
                'date': roster.date,
                'schedule': roster.schedule,
                'attendance': {
                    'department': attendance_record.department if attendance_record else None,
                    'first_in': attendance_record.first_in if attendance_record else None,
                    'last_out': attendance_record.last_out if attendance_record else None,
                    'gross_time': attendance_record.gross_time if attendance_record else None,
                    'out_of_office_time': attendance_record.out_of_office_time if attendance_record else None,
                    'out_of_office_count': attendance_record.out_of_office_count if attendance_record else None,
                    'net_office_time': attendance_record.net_office_time if attendance_record else None,
                }
            })
        return Response(results)

