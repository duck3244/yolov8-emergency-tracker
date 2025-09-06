"""
visualization.py - 시각화 모듈
지도, 그래프, 대시보드 등 다양한 시각화 기능을 제공
"""

import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cv2
import json
import webbrowser
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MapVisualizer:
    """지도 시각화 클래스"""
    
    def __init__(self):
        self.map_objects = {}
    
    def create_emergency_map(self, locations: List[Dict], center_coords: Tuple[float, float] = None,
                           zoom_start: int = 13) -> folium.Map:
        """
        긴급상황 모니터링 지도 생성
        
        Args:
            locations (List[Dict]): 위치 정보 리스트
                [{'name': str, 'lat': float, 'lon': float, 'people_count': int, 'status': str}, ...]
            center_coords (Tuple): 지도 중심 좌표 (lat, lon)
            zoom_start (int): 초기 줌 레벨
            
        Returns:
            folium.Map: 생성된 지도 객체
        """
        # 중심 좌표 계산
        if center_coords is None and locations:
            center_lat = np.mean([loc['lat'] for loc in locations])
            center_lon = np.mean([loc['lon'] for loc in locations])
            center_coords = (center_lat, center_lon)
        elif center_coords is None:
            center_coords = (37.5665, 126.9780)  # 서울시청 기본 좌표
        
        # 베이스 맵 생성
        m = folium.Map(
            location=center_coords,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # 다양한 타일 레이어 추가
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        # 각 위치에 마커 추가
        for location in locations:
            self._add_location_marker(m, location)
        
        # 히트맵 레이어 추가 (선택적)
        if len(locations) > 1:
            self._add_heatmap_layer(m, locations)
        
        # 제어 패널 추가
        folium.LayerControl().add_to(m)
        
        # 미니맵 추가
        minimap = plugins.MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # 전체화면 버튼 추가
        plugins.Fullscreen().add_to(m)
        
        # 측정 도구 추가
        plugins.MeasureControl().add_to(m)
        
        return m
    
    def _add_location_marker(self, map_obj: folium.Map, location: Dict):
        """위치 마커 추가"""
        people_count = location.get('people_count', 0)
        status = location.get('status', 'normal')
        name = location.get('name', 'Unknown')
        
        # 상태에 따른 색상 설정
        if status == 'emergency' or people_count > 50:
            color = 'red'
            icon = 'exclamation-triangle'
            prefix = 'fa'
        elif status == 'warning' or people_count > 20:
            color = 'orange'
            icon = 'warning'
            prefix = 'fa'
        else:
            color = 'green'
            icon = 'users'
            prefix = 'fa'
        
        # 마커 크기는 인원수에 비례
        radius = max(10, min(people_count * 2, 50))
        
        # 원형 마커
        folium.CircleMarker(
            location=[location['lat'], location['lon']],
            radius=radius,
            popup=self._create_popup_content(location),
            tooltip=f"{name}: {people_count}명",
            color='black',
            weight=2,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(map_obj)
        
        # 아이콘 마커
        folium.Marker(
            [location['lat'], location['lon']],
            popup=self._create_popup_content(location),
            tooltip=f"{name}: {people_count}명",
            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix=prefix
            )
        ).add_to(map_obj)
    
    def _create_popup_content(self, location: Dict) -> str:
        """팝업 내용 생성"""
        people_count = location.get('people_count', 0)
        name = location.get('name', 'Unknown')
        status = location.get('status', 'normal')
        last_updated = location.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        status_korean = {
            'normal': '정상',
            'warning': '주의',
            'emergency': '위험'
        }
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            <h4 style="margin: 0; color: #333;">{name}</h4>
            <hr style="margin: 5px 0;">
            <p><strong>현재 인원:</strong> {people_count}명</p>
            <p><strong>상태:</strong> {status_korean.get(status, status)}</p>
            <p><strong>마지막 업데이트:</strong><br>{last_updated}</p>
        </div>
        """
        return popup_html
    
    def _add_heatmap_layer(self, map_obj: folium.Map, locations: List[Dict]):
        """히트맵 레이어 추가"""
        heat_data = []
        for location in locations:
            heat_data.append([
                location['lat'],
                location['lon'],
                location.get('people_count', 1)
            ])
        
        heatmap = plugins.HeatMap(heat_data, name='인원 밀도', show=False)
        heatmap.add_to(map_obj)
    
    def save_map(self, map_obj: folium.Map, filename: str = "emergency_map.html",
                auto_open: bool = True):
        """지도를 HTML 파일로 저장"""
        map_obj.save(filename)
        print(f"지도가 '{filename}'에 저장되었습니다.")
        
        if auto_open:
            webbrowser.open(f'file://{os.path.abspath(filename)}')


class DashboardVisualizer:
    """대시보드 시각화 클래스"""
    
    def __init__(self):
        self.fig = None
    
    def create_realtime_dashboard(self, current_data: Dict, 
                                historical_data: List[Dict] = None) -> go.Figure:
        """
        실시간 대시보드 생성
        
        Args:
            current_data (Dict): 현재 데이터
            historical_data (List[Dict]): 과거 데이터 (선택적)
        
        Returns:
            plotly.graph_objects.Figure: 대시보드 Figure
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '현재 건물별 인원 현황',
                '시간별 입출입 추이',
                '주간 통계',
                '경고 레벨 분포',
                '실시간 모니터링',
                '일일 요약'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "indicator"}, {"type": "table"}]
            ]
        )
        
        # 1. 현재 건물별 인원 현황 (막대그래프)
        buildings = list(current_data.get('buildings', {}).keys())
        people_counts = list(current_data.get('buildings', {}).values())
        
        fig.add_trace(
            go.Bar(
                x=buildings,
                y=people_counts,
                name='현재 인원',
                marker_color=['red' if count > 30 else 'orange' if count > 15 else 'green' 
                             for count in people_counts]
            ),
            row=1, col=1
        )
        
        # 2. 시간별 추이 (선 그래프)
        if historical_data:
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['total_people'],
                    mode='lines+markers',
                    name='총 인원',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
        
        # 3. 주간 통계 (막대그래프)
        days = ['월', '화', '수', '목', '금', '토', '일']
        daily_avg = current_data.get('weekly_stats', [20, 25, 30, 28, 35, 15, 10])
        
        fig.add_trace(
            go.Bar(
                x=days,
                y=daily_avg,
                name='일평균 인원',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # 4. 경고 레벨 분포 (파이 차트)
        alert_levels = current_data.get('alert_distribution', {'정상': 70, '주의': 20, '위험': 10})
        
        fig.add_trace(
            go.Pie(
                labels=list(alert_levels.keys()),
                values=list(alert_levels.values()),
                hole=0.4,
                marker_colors=['green', 'orange', 'red']
            ),
            row=2, col=2
        )
        
        # 5. 실시간 모니터링 (게이지)
        current_total = sum(people_counts) if people_counts else 0
        max_capacity = current_data.get('max_capacity', 100)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_total,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "현재 총 인원"},
                delta={'reference': max_capacity * 0.7},
                gauge={
                    'axis': {'range': [None, max_capacity]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, max_capacity * 0.5], 'color': "lightgray"},
                        {'range': [max_capacity * 0.5, max_capacity * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_capacity * 0.9
                    }
                }
            ),
            row=3, col=1
        )
        
        # 6. 일일 요약 테이블
        summary_data = current_data.get('daily_summary', {
            '총 입장': 150,
            '총 퇴장': 145,
            '현재 인원': current_total,
            '최대 동시 인원': 45,
            '평균 체류시간': '2.3시간'
        })
        
        fig.add_trace(
            go.Table(
                header=dict(values=['항목', '값'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[list(summary_data.keys()), list(summary_data.values())],
                          fill_color='lavender',
                          align='left')
            ),
            row=3, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title_text="🚨 긴급상황 모니터링 대시보드",
            title_x=0.5,
            showlegend=False,
            height=1000,
            font=dict(size=12)
        )
        
        self.fig = fig
        return fig
    
    def create_analytics_dashboard(self, analytics_data: Dict) -> go.Figure:
        """분석 대시보드 생성"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '시간대별 평균 인원',
                '요일별 패턴',
                '월별 트렌드',
                '피크 시간대 분석'
            )
        )
        
        # 1. 시간대별 평균 인원
        hours = list(range(24))
        hourly_avg = analytics_data.get('hourly_average', [0] * 24)
        
        fig.add_trace(
            go.Bar(
                x=hours,
                y=hourly_avg,
                name='시간대별 평균',
                marker_color='steelblue'
            ),
            row=1, col=1
        )
        
        # 2. 요일별 패턴
        weekdays = ['월', '화', '수', '목', '금', '토', '일']
        daily_pattern = analytics_data.get('daily_pattern', [0] * 7)
        
        fig.add_trace(
            go.Scatter(
                x=weekdays,
                y=daily_pattern,
                mode='lines+markers',
                name='요일별 패턴',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # 3. 월별 트렌드
        months = ['1월', '2월', '3월', '4월', '5월', '6월',
                 '7월', '8월', '9월', '10월', '11월', '12월']
        monthly_trend = analytics_data.get('monthly_trend', [0] * 12)
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=monthly_trend,
                mode='lines+markers',
                name='월별 트렌드',
                line=dict(color='green', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # 4. 피크 시간대 분석 (히트맵)
        peak_data = analytics_data.get('peak_heatmap', np.random.rand(7, 24))
        
        fig.add_trace(
            go.Heatmap(
                z=peak_data,
                x=hours,
                y=weekdays,
                colorscale='Viridis',
                showscale=True
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📊 인원 분석 대시보드",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def save_dashboard(self, filename: str = "dashboard.html", auto_open: bool = True):
        """대시보드를 HTML 파일로 저장"""
        if self.fig:
            self.fig.write_html(filename)
            print(f"대시보드가 '{filename}'에 저장되었습니다.")
            
            if auto_open:
                webbrowser.open(f'file://{os.path.abspath(filename)}')
        else:
            print("저장할 대시보드가 없습니다.")


class ChartVisualizer:
    """차트 시각화 클래스"""
    
    def __init__(self):
        self.style = 'seaborn-v0_8'
        plt.style.use('default')  # seaborn 스타일이 없을 경우 기본 사용
        
    def plot_entry_exit_trend(self, data: List[Dict], save_path: str = None) -> plt.Figure:
        """입출입 트렌드 차트"""
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. 입장/퇴장 시계열
        ax1.plot(df['timestamp'], df['entries'], 'g-', label='입장', linewidth=2)
        ax1.plot(df['timestamp'], df['exits'], 'r-', label='퇴장', linewidth=2)
        ax1.fill_between(df['timestamp'], df['entries'], alpha=0.3, color='green')
        ax1.fill_between(df['timestamp'], df['exits'], alpha=0.3, color='red')
        ax1.set_title('입장/퇴장 추이', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 현재 건물 내 인원
        current_inside = df['entries'].cumsum() - df['exits'].cumsum()
        ax2.plot(df['timestamp'], current_inside, 'b-', linewidth=3)
        ax2.fill_between(df['timestamp'], current_inside, alpha=0.4, color='blue')
        ax2.set_title('건물 내 현재 인원', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 시간당 변화율
        hourly_change = df['entries'] - df['exits']
        colors = ['red' if x < 0 else 'green' for x in hourly_change]
        ax3.bar(df['timestamp'], hourly_change, color=colors, alpha=0.7)
        ax3.set_title('시간당 인원 변화 (입장 - 퇴장)', fontsize=14, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_occupancy_distribution(self, data: List[int], save_path: str = None) -> plt.Figure:
        """점유율 분포 히스토그램"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 히스토그램
        ax1.hist(data, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(data), color='red', linestyle='--', 
                   label=f'평균: {np.mean(data):.1f}명')
        ax1.axvline(np.median(data), color='orange', linestyle='--', 
                   label=f'중간값: {np.median(data):.1f}명')
        ax1.set_title('인원 분포', fontsize=14, fontweight='bold')
        ax1.set_xlabel('인원 수')
        ax1.set_ylabel('빈도')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 박스 플롯
        box_data = [data]
        ax2.boxplot(box_data, labels=['전체 기간'])
        ax2.set_title('인원 분포 박스 플롯', fontsize=14, fontweight='bold')
        ax2.set_ylabel('인원 수')
        ax2.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        stats_text = f"""
        최솟값: {min(data)}명
        최댓값: {max(data)}명
        평균: {np.mean(data):.1f}명
        표준편차: {np.std(data):.1f}
        """
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_heatmap(self, data: np.ndarray, x_labels: List[str], 
                    y_labels: List[str], title: str = "히트맵", 
                    save_path: str = None) -> plt.Figure:
        """히트맵 생성"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # 축 라벨 설정
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # 값 표시
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, f'{data[i, j]:.0f}',
                             ha="center", va="center", color="black")
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        fig.colorbar(im, ax=ax, label='인원 수')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class VideoVisualizer:
    """비디오 시각화 클래스"""
    
    def __init__(self):
        self.colors = {
            'person': (0, 255, 0),
            'entrance': (0, 255, 0),
            'exit': (0, 0, 255),
            'warning': (0, 255, 255),
            'danger': (0, 0, 255),
            'text_bg': (0, 0, 0),
            'text': (255, 255, 255)
        }
    
    def draw_tracking_info(self, frame: np.ndarray, tracked_objects: List,
                          areas: Dict, counts: Dict) -> np.ndarray:
        """프레임에 추적 정보 그리기"""
        result_frame = frame.copy()
        
        # 영역 그리기
        if 'entrance' in areas:
            cv2.polylines(result_frame, 
                         [np.array(areas['entrance'], np.int32)], 
                         True, self.colors['entrance'], 2)
            cv2.putText(result_frame, "ENTRANCE", 
                       tuple(areas['entrance'][0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       self.colors['entrance'], 2)
        
        if 'exit' in areas:
            cv2.polylines(result_frame, 
                         [np.array(areas['exit'], np.int32)], 
                         True, self.colors['exit'], 2)
            cv2.putText(result_frame, "EXIT", 
                       tuple(areas['exit'][0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       self.colors['exit'], 2)
        
        # 추적 객체 그리기
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 바운딩 박스
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), 
                         self.colors['person'], 2)
            
            # 중심점
            cv2.circle(result_frame, (center_x, center_y), 4, 
                      self.colors['person'], -1)
            
            # ID 라벨
            cv2.putText(result_frame, f'ID: {obj_id}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.colors['person'], 2)
        
        # 카운트 정보 패널 그리기
        self._draw_info_panel(result_frame, counts)
        
        return result_frame
    
    def _draw_info_panel(self, frame: np.ndarray, counts: Dict):
        """정보 패널 그리기"""
        panel_width = 400
        panel_height = 200
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10
        
        # 패널 배경
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['text_bg'], -1)
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['text'], 2)
        
        # 제목
        cv2.putText(frame, "MONITORING STATUS", 
                   (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   self.colors['text'], 2)
        
        # 구분선
        cv2.line(frame, (panel_x + 10, panel_y + 40), 
                (panel_x + panel_width - 10, panel_y + 40),
                self.colors['text'], 1)
        
        # 정보 표시
        info_lines = [
            f"Entered: {counts.get('entered', 0)}",
            f"Exited: {counts.get('exited', 0)}",
            f"Current Inside: {counts.get('current_inside', 0)}",
            f"Status: {counts.get('status', 'Normal')}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = panel_y + 70 + i * 25
            color = self.colors['text']
            
            # 상태에 따른 색상 변경
            if "Current Inside" in line:
                current_count = counts.get('current_inside', 0)
                if current_count > 50:
                    color = (0, 0, 255)  # 빨간색
                elif current_count > 20:
                    color = (0, 255, 255)  # 노란색
            
            cv2.putText(frame, line, (panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def create_summary_video(self, input_video_path: str, output_path: str,
                           tracking_data: List[Dict]):
        """요약 비디오 생성"""
        cap = cv2.VideoCapture(input_video_path)
        
        # 비디오 속성
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 해당 프레임의 추적 데이터 가져오기
            if frame_idx < len(tracking_data):
                frame_data = tracking_data[frame_idx]
                frame = self.draw_tracking_info(
                    frame, 
                    frame_data.get('tracked_objects', []),
                    frame_data.get('areas', {}),
                    frame_data.get('counts', {})
                )
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"처리 중: {frame_idx} 프레임")
        
        cap.release()
        out.release()
        print(f"요약 비디오가 '{output_path}'에 저장되었습니다.")


if __name__ == "__main__":
    # 테스트 코드
    print("=== 시각화 모듈 테스트 ===")
    
    # 1. 지도 시각화 테스트
    print("\n1. 지도 시각화 테스트")
    map_viz = MapVisualizer()
    
    test_locations = [
        {
            'name': '서울시청',
            'lat': 37.5665,
            'lon': 126.9780,
            'people_count': 25,
            'status': 'normal',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'name': '강남구청',
            'lat': 37.5172,
            'lon': 127.0473,
            'people_count': 45,
            'status': 'warning',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'name': '종로구청',
            'lat': 37.5735,
            'lon': 126.9788,
            'people_count': 60,
            'status': 'emergency',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    ]
    
    emergency_map = map_viz.create_emergency_map(test_locations)
    map_viz.save_map(emergency_map, "test_emergency_map.html", auto_open=False)
    
    # 2. 대시보드 시각화 테스트
    print("\n2. 대시보드 시각화 테스트")
    dashboard_viz = DashboardVisualizer()
    
    current_data = {
        'buildings': {
            '본관': 25,
            '별관': 15,
            '연구동': 35,
            '강의동': 20
        },
        'max_capacity': 150,
        'weekly_stats': [25, 30, 35, 32, 40, 20, 15],
        'alert_distribution': {'정상': 60, '주의': 30, '위험': 10},
        'daily_summary': {
            '총 입장': 180,
            '총 퇴장': 175,
            '현재 인원': 95,
            '최대 동시 인원': 105,
            '평균 체류시간': '3.2시간'
        }
    }
    
    # 과거 데이터 생성 (예제)
    historical_data = []
    base_time = datetime.now() - timedelta(hours=12)
    for i in range(144):  # 12시간, 5분 간격
        historical_data.append({
            'timestamp': (base_time + timedelta(minutes=i*5)).isoformat(),
            'total_people': np.random.randint(20, 100)
        })
    
    dashboard_fig = dashboard_viz.create_realtime_dashboard(current_data, historical_data)
    dashboard_viz.save_dashboard("test_dashboard.html", auto_open=False)
    
    # 3. 차트 시각화 테스트
    print("\n3. 차트 시각화 테스트")
    chart_viz = ChartVisualizer()
    
    # 테스트 데이터 생성
    test_trend_data = []
    for i in range(24):
        test_trend_data.append({
            'timestamp': datetime.now() - timedelta(hours=23-i),
            'entries': np.random.randint(5, 20),
            'exits': np.random.randint(3, 18)
        })
    
    trend_fig = chart_viz.plot_entry_exit_trend(test_trend_data, "test_trend.png")
    plt.close(trend_fig)
    
    # 점유율 분포 테스트
    occupancy_data = np.random.normal(30, 10, 1000).astype(int)
    occupancy_data = np.clip(occupancy_data, 0, 100)
    
    dist_fig = chart_viz.plot_occupancy_distribution(occupancy_data, "test_distribution.png")
    plt.close(dist_fig)
    
    print("\n✅ 모든 시각화 테스트 완료!")
    print("생성된 파일들:")
    print("- test_emergency_map.html")
    print("- test_dashboard.html") 
    print("- test_trend.png")
    print("- test_distribution.png")
