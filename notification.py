"""
notification_fixed.py - 완성된 알림 시스템 모듈
이메일, SMS, 웹훅 등을 통한 긴급상황 알림 시스템
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import requests
import json
import logging
from typing import List, Dict, Optional
import threading
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailNotifier:
    """이메일 알림 클래스"""

    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        """
        이메일 알림 초기화

        Args:
            smtp_server (str): SMTP 서버 주소
            smtp_port (int): SMTP 서버 포트
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = None
        self.sender_password = None
        self.is_configured = False

    def configure(self, sender_email: str, sender_password: str):
        """
        이메일 설정

        Args:
            sender_email (str): 발신자 이메일
            sender_password (str): 발신자 비밀번호 (Gmail 앱 비밀번호)
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.is_configured = True
        logger.info(f"이메일 알림이 설정되었습니다: {sender_email}")

    def send_emergency_alert(self, recipient_emails: List[str], people_count: int,
                           location_info: Dict, additional_info: str = ""):
        """
        긴급상황 알림 이메일 발송

        Args:
            recipient_emails (List[str]): 수신자 이메일 목록
            people_count (int): 현재 건물 내 인원 수
            location_info (Dict): 위치 정보 {'name': str, 'lat': float, 'lon': float}
            additional_info (str): 추가 정보
        """
        if not self.is_configured:
            logger.error("이메일이 설정되지 않았습니다.")
            return False

        try:
            # 메시지 생성
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(recipient_emails)
            msg["Subject"] = "🚨 긴급상황 알림 - 건물 내부 인원 현황"

            # 이메일 본문 생성
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = self._generate_emergency_email_body(
                people_count, location_info, current_time, additional_info
            )

            msg.attach(MIMEText(body, "html", "utf-8"))

            # SMTP 서버를 통한 이메일 발송
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                text = msg.as_string()
                server.sendmail(self.sender_email, recipient_emails, text)

            logger.info(f"긴급상황 이메일이 {len(recipient_emails)}명에게 발송되었습니다.")
            return True

        except Exception as e:
            logger.error(f"이메일 발송 실패: {e}")
            return False

    def send_periodic_report(self, recipient_emails: List[str], report_data: Dict):
        """
        주기적 리포트 이메일 발송

        Args:
            recipient_emails (List[str]): 수신자 이메일 목록
            report_data (Dict): 리포트 데이터
        """
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(recipient_emails)
            msg["Subject"] = f"📊 일일 현황 리포트 - {datetime.now().strftime('%Y-%m-%d')}"

            body = self._generate_report_email_body(report_data)
            msg.attach(MIMEText(body, "html", "utf-8"))

            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_emails, msg.as_string())

            logger.info("주기적 리포트가 발송되었습니다.")
            return True

        except Exception as e:
            logger.error(f"리포트 이메일 발송 실패: {e}")
            return False

    def _generate_emergency_email_body(self, people_count: int, location_info: Dict,
                                     timestamp: str, additional_info: str):
        """긴급상황 이메일 본문 생성"""
        status_color = "#ff4444" if people_count > 50 else "#ff8800" if people_count > 20 else "#44ff44"

        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #ff4444; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; border: 2px solid #ff4444; }}
                .info-box {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid {status_color}; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #666; }}
                .urgent {{ color: #ff0000; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 긴급상황 알림</h1>
            </div>
            <div class="content">
                <h2>건물 내부 인원 현황</h2>
                
                <div class="info-box">
                    <h3>📍 위치 정보</h3>
                    <p><strong>건물명:</strong> {location_info.get('name', 'N/A')}</p>
                    <p><strong>좌표:</strong> {location_info.get('lat', 'N/A')}, {location_info.get('lon', 'N/A')}</p>
                    <p><strong>시간:</strong> {timestamp}</p>
                </div>
                
                <div class="info-box">
                    <h3 class="urgent">👥 현재 건물 내부 인원: {people_count}명</h3>
                </div>
                
                {f'<div class="info-box"><h3>ℹ️ 추가 정보</h3><p>{additional_info}</p></div>' if additional_info else ''}
                
                <div class="info-box">
                    <h3>⚠️ 권장 조치사항</h3>
                    <ul>
                        <li>즉시 현장 확인 요청</li>
                        <li>필요시 비상 대응팀 파견</li>
                        <li>대피 경로 확보 점검</li>
                        <li>추가 모니터링 강화</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>이 메시지는 자동으로 생성된 긴급상황 알림입니다.</p>
                <p>시스템 문의: emergency-system@company.com</p>
            </div>
        </body>
        </html>
        """
        return html_body

    def _generate_report_email_body(self, report_data: Dict):
        """리포트 이메일 본문 생성"""
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2196F3; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .stats-box {{ background-color: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 일일 현황 리포트</h1>
                <p>{datetime.now().strftime('%Y년 %m월 %d일')}</p>
            </div>
            <div class="content">
                <div class="stats-box">
                    <h3>📈 전체 통계</h3>
                    <p><strong>총 입장자:</strong> {report_data.get('total_entered', 0)}명</p>
                    <p><strong>총 퇴장자:</strong> {report_data.get('total_exited', 0)}명</p>
                    <p><strong>현재 건물 내:</strong> {report_data.get('current_inside', 0)}명</p>
                    <p><strong>최대 동시 수용:</strong> {report_data.get('max_occupancy', 0)}명</p>
                </div>
                
                <div class="stats-box">
                    <h3>⏰ 시간별 현황</h3>
                    <table>
                        <tr><th>시간</th><th>입장</th><th>퇴장</th></tr>
        """

        # 시간별 데이터 추가
        hourly_data = report_data.get('hourly_stats', {})
        for hour in range(24):
            entries = hourly_data.get('hourly_entries', {}).get(hour, 0)
            exits = hourly_data.get('hourly_exits', {}).get(hour, 0)
            html_body += f"<tr><td>{hour:02d}:00</td><td>{entries}</td><td>{exits}</td></tr>"

        html_body += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        return html_body


class WebhookNotifier:
    """완성된 웹훅 알림 클래스"""

    def __init__(self):
        self.webhooks = {}  # {name: {'url': str, 'headers': dict}}
        self.default_headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Emergency-Tracker-System/1.0'
        }
        self.timeout = 10
        self.retry_count = 3

    def add_webhook(self, name: str, url: str, custom_headers: Dict = None):
        """
        웹훅 URL 추가

        Args:
            name (str): 웹훅 이름
            url (str): 웹훅 URL
            custom_headers (Dict): 커스텀 헤더
        """
        self.webhooks[name] = {
            'url': url,
            'headers': custom_headers or {}
        }
        logger.info(f"웹훅이 추가되었습니다: {name}")

    def remove_webhook(self, name: str):
        """웹훅 제거"""
        if name in self.webhooks:
            del self.webhooks[name]
            logger.info(f"웹훅이 제거되었습니다: {name}")
        else:
            logger.warning(f"존재하지 않는 웹훅: {name}")

    def list_webhooks(self) -> List[str]:
        """등록된 웹훅 목록 반환"""
        return list(self.webhooks.keys())

    def send_alert(self, webhook_name: str, alert_data: Dict):
        """
        웹훅으로 알림 전송

        Args:
            webhook_name (str): 웹훅 이름
            alert_data (Dict): 전송할 데이터

        Returns:
            bool: 전송 성공 여부
        """
        if webhook_name not in self.webhooks:
            logger.error(f"웹훅을 찾을 수 없습니다: {webhook_name}")
            return False

        webhook_config = self.webhooks[webhook_name]
        url = webhook_config['url']

        # 헤더 준비
        headers = self.default_headers.copy()
        headers.update(webhook_config.get('headers', {}))

        # 페이로드 준비
        payload = {
            'timestamp': datetime.now().isoformat(),
            'source': 'Emergency Tracking System',
            'webhook_name': webhook_name,
            **alert_data
        }

        # 재시도 로직
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code in [200, 201, 202, 204]:
                    logger.info(f"웹훅 알림 전송 성공: {webhook_name} (상태코드: {response.status_code})")
                    return True
                else:
                    logger.error(f"웹훅 알림 전송 실패: {webhook_name} (상태코드: {response.status_code})")
                    if attempt < self.retry_count - 1:
                        logger.info(f"재시도 중... ({attempt + 1}/{self.retry_count})")
                        time.sleep(1)

            except requests.exceptions.Timeout:
                logger.error(f"웹훅 전송 시간 초과: {webhook_name} (시도 {attempt + 1})")
                if attempt < self.retry_count - 1:
                    time.sleep(1)
            except requests.exceptions.ConnectionError:
                logger.error(f"웹훅 연결 오류: {webhook_name} (시도 {attempt + 1})")
                if attempt < self.retry_count - 1:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"웹훅 전송 오류 ({webhook_name}): {e}")
                break

        return False

    def send_to_all_webhooks(self, alert_data: Dict):
        """모든 등록된 웹훅으로 알림 전송"""
        results = {}
        for name in self.webhooks:
            results[name] = self.send_alert(name, alert_data)

        success_count = sum(results.values())
        total_count = len(results)
        logger.info(f"웹훅 일괄 전송 완료: {success_count}/{total_count} 성공")

        return results

    def send_emergency_alert(self, people_count: int, location_info: Dict,
                           additional_info: str = "", webhook_names: List[str] = None):
        """긴급상황 웹훅 알림 전송"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 심각도 결정
            if people_count > 50:
                severity = "HIGH"
                color = "#ff0000"
                emoji = "🔴"
            elif people_count > 20:
                severity = "MEDIUM"
                color = "#ff8800"
                emoji = "🟡"
            else:
                severity = "LOW"
                color = "#00ff00"
                emoji = "🟢"

            alert_data = {
                "alert_type": "emergency_people_count",
                "severity": severity,
                "emoji": emoji,
                "people_count": people_count,
                "location": {
                    "name": location_info.get('name', 'Unknown'),
                    "latitude": location_info.get('lat', 0),
                    "longitude": location_info.get('lon', 0)
                },
                "message": f"{emoji} 긴급상황 알림: 현재 건물 내 인원 {people_count}명",
                "additional_info": additional_info,
                "timestamp": current_time,
                "color": color,
                "actions": [
                    {
                        "text": "Google Maps에서 보기",
                        "url": f"https://maps.google.com/maps?q={location_info.get('lat', 0)},{location_info.get('lon', 0)}"
                    }
                ]
            }

            # 특정 웹훅들만 또는 전체 웹훅에 전송
            if webhook_names:
                results = {}
                for name in webhook_names:
                    if name in self.webhooks:
                        results[name] = self.send_alert(name, alert_data)
                    else:
                        logger.warning(f"존재하지 않는 웹훅: {name}")
                        results[name] = False
                return results
            else:
                return self.send_to_all_webhooks(alert_data)

        except Exception as e:
            logger.error(f"긴급상황 웹훅 알림 생성 오류: {e}")
            return {}

    def test_webhook(self, webhook_name: str):
        """웹훅 연결 테스트"""
        test_data = {
            "test": True,
            "message": "웹훅 연결 테스트입니다.",
            "test_time": datetime.now().isoformat()
        }

        logger.info(f"웹훅 테스트 시작: {webhook_name}")
        result = self.send_alert(webhook_name, test_data)

        if result:
            logger.info(f"✅ 웹훅 테스트 성공: {webhook_name}")
        else:
            logger.error(f"❌ 웹훅 테스트 실패: {webhook_name}")

        return result

    def test_all_webhooks(self):
        """모든 웹훅 연결 테스트"""
        logger.info("모든 웹훅 테스트 시작...")

        results = {}
        for name in self.webhooks:
            results[name] = self.test_webhook(name)

        success_count = sum(results.values())
        total_count = len(results)

        logger.info(f"웹훅 테스트 완료: {success_count}/{total_count} 성공")

        # 실패한 웹훅들 로그
        failed_webhooks = [name for name, success in results.items() if not success]
        if failed_webhooks:
            logger.warning(f"실패한 웹훅들: {', '.join(failed_webhooks)}")

        return results


class SlackNotifier:
    """Slack 알림 클래스"""

    def __init__(self, webhook_url: str):
        """
        Slack 알림 초기화

        Args:
            webhook_url (str): Slack Incoming Webhook URL
        """
        self.webhook_url = webhook_url

    def send_emergency_alert(self, people_count: int, location_info: Dict,
                           additional_info: str = ""):
        """긴급상황 Slack 알림 발송"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 심각도에 따른 색상 설정
            if people_count > 50:
                color = "#ff0000"  # 빨간색 - 위험
                urgency = "🔴 HIGH"
            elif people_count > 20:
                color = "#ff8800"  # 주황색 - 주의
                urgency = "🟡 MEDIUM"
            else:
                color = "#00ff00"  # 녹색 - 정상
                urgency = "🟢 LOW"

            payload = {
                "text": "🚨 긴급상황 알림",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "위치",
                                "value": f"{location_info.get('name', 'N/A')}",
                                "short": True
                            },
                            {
                                "title": "현재 인원",
                                "value": f"{people_count}명",
                                "short": True
                            },
                            {
                                "title": "심각도",
                                "value": urgency,
                                "short": True
                            },
                            {
                                "title": "발생 시간",
                                "value": current_time,
                                "short": True
                            }
                        ],
                        "footer": "Emergency Tracking System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }

            if additional_info:
                payload["attachments"][0]["fields"].append({
                    "title": "추가 정보",
                    "value": additional_info,
                    "short": False
                })

            response = requests.post(self.webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("Slack 알림 전송 성공")
                return True
            else:
                logger.error(f"Slack 알림 전송 실패: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Slack 알림 전송 오류: {e}")
            return False


class DiscordNotifier:
    """Discord 알림 클래스"""

    def __init__(self, webhook_url: str):
        """
        Discord 알림 초기화

        Args:
            webhook_url (str): Discord Webhook URL
        """
        self.webhook_url = webhook_url

    def send_emergency_alert(self, people_count: int, location_info: Dict,
                           additional_info: str = ""):
        """긴급상황 Discord 알림 발송"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 심각도에 따른 색상 설정 (Discord는 decimal 색상 사용)
            if people_count > 50:
                color = 16711680  # 빨간색
                urgency = "🔴 HIGH ALERT"
            elif people_count > 20:
                color = 16753920  # 주황색
                urgency = "🟡 MEDIUM ALERT"
            else:
                color = 65280     # 녹색
                urgency = "🟢 LOW ALERT"

            embed = {
                "title": "🚨 긴급상황 알림",
                "description": f"건물 내부 인원 현황이 보고되었습니다.",
                "color": color,
                "fields": [
                    {
                        "name": "📍 위치",
                        "value": location_info.get('name', 'N/A'),
                        "inline": True
                    },
                    {
                        "name": "👥 현재 인원",
                        "value": f"{people_count}명",
                        "inline": True
                    },
                    {
                        "name": "⚠️ 심각도",
                        "value": urgency,
                        "inline": True
                    },
                    {
                        "name": "🕐 발생 시간",
                        "value": current_time,
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "Emergency Tracking System"
                },
                "timestamp": datetime.now().isoformat()
            }

            if additional_info:
                embed["fields"].append({
                    "name": "ℹ️ 추가 정보",
                    "value": additional_info,
                    "inline": False
                })

            payload = {
                "embeds": [embed]
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)

            if response.status_code == 204:  # Discord는 204 반환
                logger.info("Discord 알림 전송 성공")
                return True
            else:
                logger.error(f"Discord 알림 전송 실패: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Discord 알림 전송 오류: {e}")
            return False


class NotificationManager:
    """통합 알림 관리자"""

    def __init__(self):
        self.email_notifier = EmailNotifier()
        self.webhook_notifier = WebhookNotifier()
        self.slack_notifier = None
        self.discord_notifier = None

        # 알림 규칙
        self.alert_rules = {
            'overcrowding_threshold': 50,
            'warning_threshold': 20,
            'emergency_contacts': [],
            'notification_interval': 300,  # 5분
            'quiet_hours': (22, 6)  # 22시~6시는 조용한 시간
        }

        # 알림 이력
        self.last_alert_time = {}
        self.alert_history = []

    def configure_email(self, sender_email: str, sender_password: str,
                       emergency_contacts: List[str]):
        """이메일 알림 설정"""
        self.email_notifier.configure(sender_email, sender_password)
        self.alert_rules['emergency_contacts'] = emergency_contacts

    def configure_slack(self, webhook_url: str):
        """Slack 알림 설정"""
        self.slack_notifier = SlackNotifier(webhook_url)

    def configure_discord(self, webhook_url: str):
        """Discord 알림 설정"""
        self.discord_notifier = DiscordNotifier(webhook_url)

    def add_custom_webhook(self, name: str, url: str, headers: Dict = None):
        """커스텀 웹훅 추가"""
        self.webhook_notifier.add_webhook(name, url, headers)

    def set_alert_rules(self, **rules):
        """알림 규칙 설정"""
        self.alert_rules.update(rules)

    def check_and_send_alerts(self, people_count: int, location_info: Dict,
                            additional_info: str = ""):
        """조건에 따른 알림 발송"""
        current_time = datetime.now()
        alert_type = self._determine_alert_type(people_count)

        if alert_type is None:
            return False

        # 중복 알림 방지
        if self._should_suppress_alert(alert_type, current_time):
            return False

        # 조용한 시간 확인
        if self._is_quiet_time(current_time) and alert_type != 'emergency':
            return False

        success = True

        # 이메일 알림
        if self.alert_rules['emergency_contacts']:
            success &= self.email_notifier.send_emergency_alert(
                self.alert_rules['emergency_contacts'],
                people_count,
                location_info,
                additional_info
            )

        # Slack 알림
        if self.slack_notifier:
            success &= self.slack_notifier.send_emergency_alert(
                people_count, location_info, additional_info
            )

        # Discord 알림
        if self.discord_notifier:
            success &= self.discord_notifier.send_emergency_alert(
                people_count, location_info, additional_info
            )

        # 커스텀 웹훅 알림
        if self.webhook_notifier.list_webhooks():
            webhook_results = self.webhook_notifier.send_emergency_alert(
                people_count, location_info, additional_info
            )
            success &= any(webhook_results.values())

        # 알림 이력 저장
        if success:
            self.last_alert_time[alert_type] = current_time
            self.alert_history.append({
                'timestamp': current_time.isoformat(),
                'alert_type': alert_type,
                'people_count': people_count,
                'location': location_info.get('name', 'N/A'),
                'additional_info': additional_info
            })

        return success

    def _determine_alert_type(self, people_count: int) -> Optional[str]:
        """알림 유형 결정"""
        if people_count >= self.alert_rules['overcrowding_threshold']:
            return 'emergency'
        elif people_count >= self.alert_rules['warning_threshold']:
            return 'warning'
        return None

    def _should_suppress_alert(self, alert_type: str, current_time: datetime) -> bool:
        """중복 알림 억제 확인"""
        if alert_type in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[alert_type]).seconds
            return time_diff < self.alert_rules['notification_interval']
        return False

    def _is_quiet_time(self, current_time: datetime) -> bool:
        """조용한 시간 확인"""
        current_hour = current_time.hour
        quiet_start, quiet_end = self.alert_rules['quiet_hours']

        if quiet_start > quiet_end:  # 밤을 넘어가는 경우 (예: 22시~6시)
            return current_hour >= quiet_start or current_hour < quiet_end
        else:
            return quiet_start <= current_hour < quiet_end

    def send_daily_report(self, report_data: Dict):
        """일일 리포트 전송"""
        if self.alert_rules['emergency_contacts']:
            return self.email_notifier.send_periodic_report(
                self.alert_rules['emergency_contacts'],
                report_data
            )
        return False

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """알림 이력 반환"""
        return self.alert_history[-limit:]

    def test_all_notifications(self, people_count: int = 25,
                              location_info: Dict = None):
        """모든 알림 시스템 테스트"""
        if location_info is None:
            location_info = {'name': 'Test Building', 'lat': 37.5665, 'lon': 126.9780}

        logger.info("알림 시스템 통합 테스트 시작...")

        results = {
            'email': False,
            'slack': False,
            'discord': False,
            'webhooks': {}
        }

        # 이메일 테스트
        if self.alert_rules['emergency_contacts']:
            results['email'] = self.email_notifier.send_emergency_alert(
                self.alert_rules['emergency_contacts'],
                people_count,
                location_info,
                "이것은 알림 시스템 테스트입니다."
            )

        # Slack 테스트
        if self.slack_notifier:
            results['slack'] = self.slack_notifier.send_emergency_alert(
                people_count, location_info, "Slack 알림 테스트"
            )

        # Discord 테스트
        if self.discord_notifier:
            results['discord'] = self.discord_notifier.send_emergency_alert(
                people_count, location_info, "Discord 알림 테스트"
            )

        # 웹훅 테스트
        if self.webhook_notifier.list_webhooks():
            results['webhooks'] = self.webhook_notifier.test_all_webhooks()

        # 결과 요약
        total_tests = 0
        successful_tests = 0

        if self.alert_rules['emergency_contacts']:
            total_tests += 1
            if results['email']:
                successful_tests += 1

        if self.slack_notifier:
            total_tests += 1
            if results['slack']:
                successful_tests += 1

        if self.discord_notifier:
            total_tests += 1
            if results['discord']:
                successful_tests += 1

        webhook_success = sum(results['webhooks'].values()) if results['webhooks'] else 0
        webhook_total = len(results['webhooks']) if results['webhooks'] else 0
        total_tests += webhook_total
        successful_tests += webhook_success

        logger.info(f"알림 시스템 테스트 완료: {successful_tests}/{total_tests} 성공")

        if successful_tests == total_tests and total_tests > 0:
            logger.info("✅ 모든 알림 시스템 테스트 성공!")
        else:
            logger.warning(f"⚠️ 일부 알림 시스템 테스트 실패 ({total_tests - successful_tests}개)")

        return results

    def get_notification_status(self) -> Dict:
        """알림 시스템 상태 반환"""
        return {
            'email_configured': self.email_notifier.is_configured,
            'emergency_contacts_count': len(self.alert_rules['emergency_contacts']),
            'slack_configured': self.slack_notifier is not None,
            'discord_configured': self.discord_notifier is not None,
            'webhook_count': len(self.webhook_notifier.list_webhooks()),
            'alert_rules': self.alert_rules,
            'alert_history_count': len(self.alert_history)
        }


class PeriodicReporter:
    """주기적 리포트 생성기"""

    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager
        self.reporting_thread = None
        self.stop_reporting = False
        self.report_schedule = {}  # {time: report_type}

    def add_daily_report(self, report_time: str = "09:00"):
        """일일 리포트 스케줄 추가"""
        self.report_schedule[report_time] = 'daily'
        logger.info(f"일일 리포트 스케줄 추가: 매일 {report_time}")

    def add_weekly_report(self, report_time: str = "09:00", weekday: int = 0):
        """주간 리포트 스케줄 추가 (0=월요일)"""
        schedule_key = f"{weekday}_{report_time}"
        self.report_schedule[schedule_key] = 'weekly'
        logger.info(f"주간 리포트 스케줄 추가: 매주 {['월','화','수','목','금','토','일'][weekday]}요일 {report_time}")

    def start_scheduled_reporting(self):
        """스케줄된 리포트 시작"""
        def report_worker():
            while not self.stop_reporting:
                current_time = datetime.now()
                current_time_str = current_time.strftime("%H:%M")
                current_weekday = current_time.weekday()

                # 일일 리포트 확인
                if current_time_str in self.report_schedule:
                    if self.report_schedule[current_time_str] == 'daily':
                        self._send_daily_report()
                        time.sleep(60)  # 1분 대기로 중복 발송 방지

                # 주간 리포트 확인
                weekly_key = f"{current_weekday}_{current_time_str}"
                if weekly_key in self.report_schedule:
                    if self.report_schedule[weekly_key] == 'weekly':
                        self._send_weekly_report()
                        time.sleep(60)

                time.sleep(30)  # 30초마다 확인

        self.reporting_thread = threading.Thread(target=report_worker)
        self.reporting_thread.daemon = True
        self.reporting_thread.start()
        logger.info("주기적 리포트 스케줄링 시작")

    def stop_scheduled_reporting(self):
        """스케줄된 리포트 중지"""
        self.stop_reporting = True
        if self.reporting_thread:
            self.reporting_thread.join()
        logger.info("주기적 리포트 스케줄링 중지")

    def _send_daily_report(self):
        """일일 리포트 발송"""
        # 실제 구현시에는 데이터베이스나 파일에서 실제 데이터를 가져와야 함
        report_data = {
            'total_entered': 150,
            'total_exited': 145,
            'current_inside': 5,
            'max_occupancy': 25,
            'hourly_stats': {
                'hourly_entries': {9: 20, 12: 30, 17: 25},
                'hourly_exits': {12: 15, 17: 40, 18: 20}
            }
        }

        success = self.notification_manager.send_daily_report(report_data)
        if success:
            logger.info("일일 리포트 발송 완료")
        else:
            logger.error("일일 리포트 발송 실패")

    def _send_weekly_report(self):
        """주간 리포트 발송"""
        # 주간 데이터 집계
        report_data = {
            'total_entered': 1050,
            'total_exited': 1045,
            'current_inside': 5,
            'max_occupancy': 45,
            'daily_averages': {
                'monday': 180,
                'tuesday': 165,
                'wednesday': 170,
                'thursday': 175,
                'friday': 190,
                'saturday': 85,
                'sunday': 85
            }
        }

        success = self.notification_manager.send_daily_report(report_data)
        if success:
            logger.info("주간 리포트 발송 완료")
        else:
            logger.error("주간 리포트 발송 실패")


if __name__ == "__main__":
    # 테스트 코드

    # 알림 관리자 초기화
    notification_manager = NotificationManager()

    # 이메일 설정 (실제 사용시 유효한 정보 필요)
    notification_manager.configure_email(
        sender_email="test@gmail.com",
        sender_password="app_password",
        emergency_contacts=["emergency1@company.com", "emergency2@company.com"]
    )

    # 웹훅 설정
    notification_manager.add_custom_webhook(
        "teams",
        "https://outlook.office.com/webhook/...",
        {"Authorization": "Bearer token123"}
    )
    notification_manager.add_custom_webhook(
        "custom_api",
        "https://api.company.com/alerts"
    )

    # Slack 설정 (실제 사용시 유효한 URL 필요)
    # notification_manager.configure_slack("https://hooks.slack.com/...")

    # Discord 설정 (실제 사용시 유효한 URL 필요)
    # notification_manager.configure_discord("https://discord.com/api/webhooks/...")

    # 알림 규칙 설정
    notification_manager.set_alert_rules(
        overcrowding_threshold=30,
        warning_threshold=15,
        notification_interval=60  # 1분
    )

    # 테스트 위치 정보
    test_location = {
        'name': '서울시청',
        'lat': 37.5665,
        'lon': 126.9780
    }

    # 다양한 시나리오 테스트
    print("=== 완성된 알림 시스템 테스트 ===")

    # 1. 알림 시스템 상태 확인
    print("\n1. 알림 시스템 상태")
    status = notification_manager.get_notification_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 2. 웹훅 테스트
    print("\n2. 웹훅 연결 테스트")
    webhook_test_results = notification_manager.webhook_notifier.test_all_webhooks()
    for webhook, result in webhook_test_results.items():
        print(f"  {webhook}: {'✅ 성공' if result else '❌ 실패'}")

    # 3. 통합 알림 테스트
    print("\n3. 통합 알림 시스템 테스트")
    test_results = notification_manager.test_all_notifications(25, test_location)
    print(f"  테스트 결과: {test_results}")

    # 4. 긴급상황 시뮬레이션
    print("\n4. 긴급상황 시뮬레이션")
    scenarios = [
        (10, "정상 상황"),
        (20, "경고 상황"),
        (40, "긴급 상황")
    ]

    for people_count, description in scenarios:
        print(f"\n  {description} 테스트 ({people_count}명)")
        result = notification_manager.check_and_send_alerts(
            people_count, test_location, f"{description} 시뮬레이션"
        )
        print(f"  결과: {'알림 발송됨' if result else '알림 없음'}")

    # 5. 알림 이력 확인
    print("\n5. 알림 이력")
    history = notification_manager.get_alert_history()
    for entry in history:
        print(f"  - {entry['timestamp']}: {entry['alert_type']} ({entry['people_count']}명)")

    print("\n✅ 완성된 알림 시스템 테스트 완료!")