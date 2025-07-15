# **LangGraph를 활용한 프로액티브 대화형 에이전트 아키텍처 설계**

## **제 1부: 패러다임의 전환 - 반응형에서 프로액티브 실행으로**

대화형 AI 에이전트 개발의 전통적인 패러다임은 사용자의 입력에 응답하는
반응형(reactive) 모델에 기반합니다. 하지만 사용자 경험을 한 단계
끌어올리고 비즈니스 가치를 극대화하기 위해서는 AI가 먼저 대화를 시작하는
프로액티브(proactive) 모델로의 전환이 필수적입니다. 본 보고서는
LangGraph를 사용하여 사용자의 요청을 기다리는 대신, 특정 조건이나
이벤트에 따라 먼저 말을 거는 지능형 챗봇을 구현하는 방법론과 아키텍처를
심도 있게 분석합니다. 이를 위해 프로액티브 실행의 근본적인 개념적
변화부터 시작하여, 실제 운영 환경에 적용 가능한 두 가지 핵심 아키텍처
패턴, 그리고 그래프 내부의 논리 설계와 운영을 위한 필수 고려사항까지
포괄적으로 다룹니다.

### **1.1 LangGraph 실행의 촉매: 비활성 상태 기계** {#langgraph-실행의-촉매-비활성-상태-기계}

LangGraph의 핵심은 상태 기계(state machine)를 생성하는 것입니다.^1^ 이
상태 기계는 본질적으로 비활성(inert) 상태로 존재하며, 외부로부터의
호출(invocation)이 있기 전까지는 어떠한 동작도 수행하지 않습니다.
대부분의 LangGraph 공식 튜토리얼은 이 개념을 명확하게 보여줍니다.

while True: 루프 안에서 사용자의 콘솔 입력을 기다리는 input() 함수를
사용하고, 그 입력을 받아 graph.stream()이나 graph.invoke()를 호출하는
구조가 바로 그것입니다.^3^ 이 구조는 근본적으로 사용자의 행동에 의존하는
반응형 패턴을 정의합니다.

프로액티브 패러다임으로의 전환은 바로 이 \'실행의 촉매\'를 내부(사용자
입력)에서 외부로 옮기는 것에서 시작됩니다. 즉, 그래프를 실행시키는
트리거(trigger)를 더 이상 사용자의 메시지에 한정하지 않고, API 호출,
데이터베이스 상태 변경, 메시지 큐의 이벤트, 또는 정해진 시간과 같은 외부
시스템의 이벤트로 확장하는 것입니다. LangGraph로 컴파일된 에이전트
자체는 호출 가능한(callable) 상태 저장 워크플로우로서 그 본질이 변하지
않으며, 무엇이 그것을 호출하는지에 따라 반응형이 될 수도, 프로액티브가
될 수도 있습니다.

이러한 설계의 가장 중요한 의의는 \*\*트리거와 로직의 분리(Decoupling of
Trigger and Logic)\*\*에 있습니다. 프로액티브 동작을 구현하는 메커니즘은
에이전트의 내부 추론 로직과 완전히 분리된 별개의 아키텍처 구성
요소입니다. 이는 강력한 관심사 분리(Separation of Concerns) 원칙을
실현하며, 일단 한 번 잘 설계된 핵심 에이전트는 다양한 상호작용
시나리오에 걸쳐 높은 재사용성을 갖게 됩니다. 예를 들어, 사용자 문의에
답변하는 동일한 에이전트 로직이 웹사이트 채팅창에서는 사용자의 메시지에
의해 트리거되고, 모바일 앱에서는 사용자가 특정 버튼을 눌렀을 때 발생하는
이벤트를 통해 프로액티브하게 먼저 말을 거는 데 사용될 수 있습니다.
결과적으로, \"어떻게 LangGraph를 프로액티브하게 만드는가?\"라는 질문에
대한 해답은 새로운 LangGraph 함수를 찾는 것이 아니라, 기존의 invoke
함수를 호출하는 외부 시스템을 어떻게 설계하고 배포할 것인가에 대한
아키텍처적 접근에서 찾아야 합니다.

### **1.2 프로액티브 상태 및 진입점 설계** {#프로액티브-상태-및-진입점-설계}

그래프 실행의 성격은 전적으로 graph.invoke()에 전달되는 초기
상태(initial state) 딕셔너리에 의해 결정됩니다. 이 초기 상태는 단순한
입력값을 넘어 전체 상호작용의 방향성을 결정하는 \'씨앗\'과 같은 역할을
합니다. 반응형 챗봇의 경우, 초기 상태는 일반적으로 사용자의 첫 메시지를
담고 있습니다 (예: {\"messages\": \[HumanMessage(\...)\]}).^6^ 하지만
프로액티브 챗봇은 시작점에 사용자 메시지가 존재하지 않으므로, \'왜\' 이
대화가 시작되었는지에 대한 \*\*트리거의 맥락(context of the
trigger)\*\*을 초기 상태에 담아주어야 합니다.

이를 위해 LangGraph가 제공하는 유연한 상태 정의 기능을 활용하여,
프로액티브와 반응형 시나리오를 모두 수용할 수 있는 커스텀 상태 스키마를
설계하는 것이 중요합니다. typing.TypedDict를 사용하면 복잡하고 명시적인
상태 구조를 만들 수 있습니다.^8^

다음은 프로액티브 상호작용을 위한 상태 스키마의 설계 예시입니다.

> Python

from typing import TypedDict, Annotated, Optional, List, Literal  
from langgraph.graph.message import add_messages  
from langchain_core.messages import BaseMessage  
  
\# 프로액티브 트리거의 유형과 메타데이터를 정의  
class ProactiveTriggerContext(TypedDict, total=False):  
\"\"\"프로액티브 실행의 원인이 되는 이벤트 정보를 담는 구조체\"\"\"  
trigger_event_type: Literal\[\'user_signup\', \'post_purchase\',
\'daily_briefing\', \'cart_abandonment\'\]  
trigger_event_id: str \# 동일 이벤트 중복 처리를 방지하기 위한 고유 ID  
user_id: str \# 개인화된 상호작용을 위한 사용자 ID  
metadata: dict \# 상품 ID, 주문 번호 등 추가적인 맥락 정보  
  
\# 전체 그래프의 상태를 정의하는 메인 스키마  
class AgentState(TypedDict):  
\"\"\"그래프의 전체 상태를 정의하는 스키마.  
  
Attributes:  
messages: 대화 기록을 저장. add_messages 리듀서를 통해 메시지가
누적됨.  
trigger_context: 프로액티브 실행 시 트리거 정보를 담는 필드.  
user_profile: user_id를 기반으로 조회된 사용자 프로필 정보.  
\"\"\"  
messages: Annotated, add_messages\]  
trigger_context: Optional  
user_profile: Optional\[dict\]

이 AgentState 스키마는 대화 기록을 관리하는 messages 필드와 더불어,
프로액티브 실행의 맥락을 담는 trigger_context라는 선택적 필드를
포함합니다. 그래프의 실행은 이 trigger_context 필드의 존재 여부에 따라
완전히 다른 경로를 따르게 됩니다. 만약 초기 상태에 trigger_context가
포함되어 있다면, 그래프는 프로액티브 로직을 담당하는 노드에서 실행을
시작해야 합니다. 이를 위해
workflow.set_entry_point(\"proactive_initiator\")와 같이 그래프의
진입점(entry point)을 명시적으로 설정하여, 초기 실행 흐름을 해당 노드로
보낼 수 있습니다.^1^ 반면,

trigger_context 없이 messages만 포함된 상태로 호출될 경우, 이는 일반적인
반응형 대화로 간주되어 다른 진입점이나 로직을 따르도록 설계할 수
있습니다. 이처럼 초기 상태 객체의 구조와 내용은 그래프의 동작을 제어하는
가장 강력하고 핵심적인 메커니즘이 됩니다.

## **제 2부: 프로액티브 트리거를 위한 핵심 아키텍처 패턴**

프로액티브 대화를 촉발시키는 방법은 크게 두 가지 아키텍처 패턴으로 나눌
수 있습니다: 특정 이벤트에 실시간으로 반응하는 **이벤트
기반(Event-Driven) API 패턴**과 정해진 시간에 주기적으로 실행되는 **시간
기반(Time-Based) 스케줄러 패턴**입니다. 각 패턴은 서로 다른 비즈니스
요구사항과 기술적 환경에 적합하므로, 구현하고자 하는 서비스의 특성에
맞춰 신중하게 선택해야 합니다.

### **2.1 패턴 I: API를 통한 이벤트 기반 프로액티브 상호작용 (FastAPI 활용)** {#패턴-i-api를-통한-이벤트-기반-프로액티브-상호작용-fastapi-활용}

이 패턴은 사용자의 회원 가입, 상품 구매, 장바구니 이탈, 고객 지원 티켓
생성과 같이 예측 불가능하고 비동기적으로 발생하는 특정 이벤트에
실시간으로 반응해야 할 때 이상적입니다. 외부 시스템(예: 웹 애플리케이션
백엔드, CRM 시스템)이 특정 이벤트를 감지했을 때, 미리 정의된 API
엔드포인트를 호출하여 LangGraph 에이전트의 실행을 트리거하는 구조입니다.

#### **2.1.1 아키텍처 개요** {#아키텍처-개요}

본 아키텍처에서는 Python의 고성능 웹 프레임워크인 FastAPI를 사용하여
외부 요청을 수신하는 게이트웨이 서버를 구축합니다.^11^ 이 서버는

/api/v1/trigger/proactive-chat과 같은 보안이 적용된 엔드포인트를 외부에
노출합니다. 외부 시스템은 이 엔드포인트로 이벤트의 맥락을 담은 JSON
페이로드(payload)를 전송합니다. FastAPI 애플리케이션은 이 요청을 받아
유효성을 검증하고, 이를 기반으로 프로액티브 실행을 위한 초기 AgentState
딕셔너리를 구성한 뒤, LangGraph 에이전트를 비동기적으로
호출(graph.ainvoke())합니다. 비동기 호출은 다수의 동시 요청을 효율적으로
처리하는 데 필수적입니다.^15^

#### **2.1.2 전체 구현 예제** {#전체-구현-예제}

다음은 FastAPI를 사용하여 이벤트 기반 프로액티브 챗봇을 구현하는 전체
코드 예제입니다. 프로젝트는 app, agent, config 디렉토리로 구조화하여
관리의 용이성을 높입니다.

**프로젝트 구조:**

proactive-fastapi-agent/  
├── app/  
│ ├── \_\_init\_\_.py  
│ ├── main.py \# FastAPI 애플리케이션 및 엔드포인트 정의  
│ └── models.py \# Pydantic 모델 (요청/응답 스키마)  
├── agent/  
│ ├── \_\_init\_\_.py  
│ ├── graph.py \# LangGraph 그래프 정의  
│ └── state.py \# AgentState 스키마 정의  
├──.env \# 환경 변수 (API 키 등)  
└── requirements.txt \# 의존성 패키지 목록

**1. 의존성 패키지 (requirements.txt):**

fastapi  
uvicorn\[standard\]  
pydantic  
python-dotenv  
langgraph  
langchain-openai  
langchain

**2. 상태 스키마 정의 (agent/state.py):**

> Python

from typing import TypedDict, Annotated, Optional, List, Literal  
from langgraph.graph.message import add_messages  
from langchain_core.messages import BaseMessage  
  
class ProactiveTriggerContext(TypedDict, total=False):  
trigger_event_type: Literal\[\'user_signup\', \'post_purchase\'\]  
trigger_event_id: str  
user_id: str  
metadata: dict  
  
class AgentState(TypedDict):  
messages: Annotated, add_messages\]  
trigger_context: Optional  
user_profile: Optional\[dict\]

**3. LangGraph 그래프 정의 (agent/graph.py):**

*이 파일의 상세 로직은 제 3부에서 자세히 다룹니다. 여기서는 기본적인
구조만 정의합니다.*

> Python

from langgraph.graph import StateGraph, END  
from langgraph.checkpoint.memory import MemorySaver  
from agent.state import AgentState  
\# 상세 노드 함수들은 제 3부에서 정의  
from.nodes import proactive_initiator_node, wait_for_user_response_node,
main_conversation_node  
  
def create_graph():  
\"\"\"프로액티브 상호작용을 위한 LangGraph 그래프를 생성하고
컴파일합니다.\"\"\"  
  
\# 메모리 기반 checkpointer 설정 (운영 환경에서는 Redis, Postgres 등
사용)  
checkpointer = MemorySaver()  
  
workflow = StateGraph(AgentState)  
  
\# 노드 추가  
workflow.add_node(\"proactive_initiator\", proactive_initiator_node)  
workflow.add_node(\"wait_for_user\", wait_for_user_response_node)  
workflow.add_node(\"conversation\", main_conversation_node)  
  
\# 진입점 설정  
workflow.set_entry_point(\"proactive_initiator\")  
  
\# 엣지 연결  
workflow.add_edge(\"proactive_initiator\", \"wait_for_user\")  
\# \'wait_for_user\' 노드는 interrupt를 통해 멈추고, 재개 시
\'conversation\'으로 이어짐  
workflow.add_edge(\"conversation\", END) \# 간단한 예시를 위해 바로
종료  
  
\# 그래프 컴파일  
return workflow.compile(checkpointer=checkpointer)  
  
\# 애플리케이션 전역에서 사용할 그래프 인스턴스  
graph_app = create_graph()

**4. Pydantic 모델 정의 (app/models.py):**

> Python

from pydantic import BaseModel, Field  
from typing import Literal, Optional, Dict, Any  
  
class ProactiveTriggerRequest(BaseModel):  
\"\"\"프로액티브 트리거 API의 요청 본문 스키마\"\"\"  
event_type: Literal\[\'user_signup\', \'post_purchase\'\] = Field(\...,
description=\"트리거 이벤트의 유형\")  
event_id: str = Field(\..., description=\"이벤트의 고유 ID (멱등성
보장용)\")  
user_id: str = Field(\..., description=\"대상 사용자의 ID\")  
metadata: Optional\] = Field(None, description=\"이벤트 관련 추가 데이터
(예: 상품 정보)\")  
thread_id: str = Field(\..., description=\"대화를 추적하기 위한 스레드
ID\")  
  
class ProactiveTriggerResponse(BaseModel):  
\"\"\"프로액티브 트리거 API의 응답 스키마\"\"\"  
status: str = \"triggered\"  
thread_id: str  
initial_message: str

**5. FastAPI 애플리케이션 (app/main.py):**

> Python

from fastapi import FastAPI, HTTPException, Body  
from app.models import ProactiveTriggerRequest,
ProactiveTriggerResponse  
from agent.graph import graph_app  
from agent.state import AgentState, ProactiveTriggerContext  
import uuid  
  
app = FastAPI(  
title=\"Proactive Agent API\",  
description=\"LangGraph 에이전트의 프로액티브 실행을 트리거하는 API\",  
version=\"1.0.0\"  
)  
  
@app.post(\"/api/v1/trigger/proactive-chat\",
response_model=ProactiveTriggerResponse)  
async def trigger_proactive_chat(request: ProactiveTriggerRequest =
Body(\...)):  
\"\"\"  
외부 이벤트에 기반하여 프로액티브 대화를 시작합니다.  
- 요청 본문을 기반으로 초기 AgentState를 구성합니다.  
- LangGraph를 비동기적으로 호출하여 대화를 시작합니다.  
\"\"\"  
thread_id = request.thread_id  
  
\# LangGraph 실행을 위한 설정  
config = {\"configurable\": {\"thread_id\": thread_id}}  
  
\# 초기 상태 구성  
initial_state = AgentState(  
messages=,  
trigger_context=ProactiveTriggerContext(  
trigger_event_type=request.event_type,  
trigger_event_id=request.event_id,  
user_id=request.user_id,  
metadata=request.metadata or {}  
),  
user_profile=None \# 실제로는 DB에서 user_id로 프로필을 조회해야 함  
)  
  
try:  
\# LangGraph 비동기 호출  
\# astream_events는 더 세분화된 제어를 제공하지만, 여기서는 ainvoke로
최종 상태를 받음  
final_state = await graph_app.ainvoke(initial_state, config=config)  
  
\# 첫 번째 AI 메시지 추출  
initial_ai_message = \"\"  
if final_state and final_state.get(\"messages\"):  
\# \'wait_for_user\' 노드에서 interrupt 되기 전의 마지막 메시지가 AI의
첫 메시지  
ai_messages = \[msg for msg in final_state\[\"messages\"\] if msg.type
== \"ai\"\]  
if ai_messages:  
initial_ai_message = ai_messages.content  
  
return ProactiveTriggerResponse(  
thread_id=thread_id,  
initial_message=initial_ai_message  
)  
except Exception as e:  
\# 실제 운영 환경에서는 더 상세한 로깅 필요  
raise HTTPException(status_code=500, detail=f\"Failed to trigger
proactive chat: {str(e)}\")  
  
\# 애플리케이션 실행 (개발용)  
\# uvicorn app.main:app \--reload

이 아키텍처는 확장성과 유지보수성이 뛰어납니다. 새로운 프로액티브
시나리오(예: \'cart_abandonment\')가 추가될 경우,
ProactiveTriggerRequest 모델의 Literal 타입에 새 이벤트 타입을 추가하고,
agent/graph.py의 proactive_initiator_node에서 해당 이벤트를 처리하는
로직만 추가하면 됩니다. API 게이트웨이를 통해 트리거를 중앙에서
관리하므로, 다양한 마이크로서비스나 외부 파트너 시스템과의 연동이
용이합니다.

### **2.2 패턴 II: 스케줄러를 통한 시간 기반 프로액티브 상호작용 (APScheduler 활용)** {#패턴-ii-스케줄러를-통한-시간-기반-프로액티브-상호작용-apscheduler-활용}

이 패턴은 매일 아침 뉴스 브리핑, 주간 성과 보고서 발송, 예약 시간 알림,
주기적인 사용자 안부 확인 등 정해진 시간에 반복적으로 작업을 수행해야 할
때 적합합니다. Python의 강력한 스케줄링 라이브러리인 APScheduler를
사용하여 LangGraph 에이전트의 실행을 자동화합니다.

#### **2.2.1 아키텍처 개요** {#아키텍처-개요-1}

이 구조에서는 APScheduler의 BackgroundScheduler를 사용하여 메인
애플리케이션 스레드를 차단하지 않고 백그라운드에서 스케줄링 작업을
실행합니다.^16^ 스케줄러는

cron이나 interval 같은 트리거를 사용하여 지정된 시간에 특정 함수(job
function)를 호출하도록 설정됩니다.^17^ 이 잡 함수는 프로액티브 대화를
시작하는 데 필요한 모든 로직을 포함합니다. 예를 들어, 데이터베이스를
쿼리하여 오늘 알림을 받아야 할 모든 사용자를 조회하고, 각 사용자에 대해
적절한

trigger_context를 담은 초기 AgentState를 구성한 뒤, graph.invoke()를
호출하여 대화를 시작합니다.

이 방식은 LangGraph Platform에서 제공하는 상용 기능인 Cron Jobs와 유사한
기능을 오픈소스 라이브러리만으로 구현할 수 있게 해주는 강력한
대안입니다.^18^

#### **2.2.2 전체 구현 예제** {#전체-구현-예제-1}

다음은 매일 오전 9시에 모든 활성 사용자에게 \"데일리 브리핑\"을 보내는
프로액티브 챗봇을 APScheduler로 구현하는 예제입니다.

**프로젝트 구조:**

proactive-scheduler-agent/  
├── agent/  
│ ├── \_\_init\_\_.py  
│ ├── graph.py \# LangGraph 그래프 정의 (패턴 I과 공유 가능)  
│ └── state.py \# AgentState 스키마 정의 (패턴 I과 공유 가능)  
├── scheduler/  
│ ├── \_\_init\_\_.py  
│ └── jobs.py \# 스케줄링될 잡 함수 정의  
├── main.py \# 스케줄러 초기화 및 실행  
├──.env  
└── requirements.txt

**1. 의존성 패키지 (requirements.txt):**

apscheduler  
python-dotenv  
langgraph  
langchain-openai  
langchain  
\# 데이터베이스 연동을 위한 psycopg2-binary 등 추가 가능

**2. 스케줄링 잡 함수 정의 (scheduler/jobs.py):**

> Python

from agent.graph import graph_app  
from agent.state import AgentState, ProactiveTriggerContext  
import uuid  
import logging  
  
\# 가상의 사용자 데이터베이스 조회 함수  
def get_active_users_for_daily_briefing():  
\"\"\"DB에서 데일리 브리핑을 수신할 활성 사용자 목록을
조회합니다.\"\"\"  
\# 실제 구현에서는 데이터베이스에 연결하여 사용자 목록을 가져옵니다.  
logging.info(\"Fetching active users from the database\...\")  
return  
  
def trigger_daily_briefing_job():  
\"\"\"  
매일 실행되는 잡: 모든 활성 사용자에게 데일리 브리핑 대화를
시작합니다.  
\"\"\"  
logging.info(\"Starting daily briefing job\...\")  
active_users = get_active_users_for_daily_briefing()  
  
for user in active_users:  
thread_id = f\"daily_briefing\_{user\[\'user_id\'\]}\_{uuid.uuid4()}\"  
config = {\"configurable\": {\"thread_id\": thread_id}}  
  
initial_state = AgentState(  
messages=,  
trigger_context=ProactiveTriggerContext(  
trigger_event_type=\'daily_briefing\',  
trigger_event_id=str(uuid.uuid4()), \# 매번 고유한 이벤트 ID 생성  
user_id=user\[\'user_id\'\],  
metadata={\"user_name\": user\[\'name\'\]}  
),  
user_profile=user \# 사용자 정보를 프로필로 전달  
)  
  
try:  
logging.info(f\"Triggering proactive chat for user {user\[\'user_id\'\]}
on thread {thread_id}\")  
\# invoke는 동기적으로 실행되므로, 많은 사용자 처리 시 비동기나 별도
워커 고려 필요  
graph_app.invoke(initial_state, config=config)  
logging.info(f\"Successfully triggered chat for user
{user\[\'user_id\'\]}\")  
except Exception as e:  
logging.error(f\"Failed to trigger chat for user {user\[\'user_id\'\]}:
{e}\")

**3. 스케줄러 실행 (main.py):**

> Python

from apscheduler.schedulers.background import BackgroundScheduler  
from scheduler.jobs import trigger_daily_briefing_job  
import time  
import logging  
import os  
from dotenv import load_dotenv  
  
\# 로깅 및 환경 변수 설정  
load_dotenv()  
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s -
%(levelname)s - %(message)s\')  
  
def main():  
\"\"\"스케줄러를 초기화하고 실행합니다.\"\"\"  
scheduler = BackgroundScheduler(timezone=\'Asia/Seoul\')  
  
\# 매일 오전 9시에 trigger_daily_briefing_job 함수를 실행하도록 잡
추가  
scheduler.add_job(  
trigger_daily_briefing_job,  
trigger=\'cron\',  
hour=9,  
minute=0,  
id=\'daily_briefing_job\', \# 잡의 고유 ID  
replace_existing=True  
)  
  
logging.info(\"Scheduler started. Press Ctrl+C to exit.\")  
scheduler.start()  
  
try:  
\# 메인 스레드가 종료되지 않도록 유지하여 백그라운드 스케줄러가 계속
실행되게 함  
while True:  
time.sleep(1)  
except (KeyboardInterrupt, SystemExit):  
\# 애플리케이션 종료 시 스케줄러를 안전하게 종료  
logging.info(\"Shutting down scheduler\...\")  
scheduler.shutdown()  
logging.info(\"Scheduler shut down successfully.\")  
  
if \_\_name\_\_ == \"\_\_main\_\_\":  
main()

이 아키텍처는 정기적인 배치(batch)성 작업을 자동화하는 데 매우
효과적입니다. main.py를 실행하면, BackgroundScheduler가 별도의
스레드에서 동작을 시작하고, 지정된 시간(매일 오전 9시)이 되면
trigger_daily_briefing_job 함수를 실행합니다. 이 함수는 데이터베이스에서
대상 사용자를 가져와 각각에 대해 LangGraph 워크플로우를 호출함으로써
프로액티브 대화를 시작합니다. while True: time.sleep(1) 루프는 메인
프로그램이 바로 종료되는 것을 막고, 스케줄러가 계속해서 동작할 수 있는
환경을 제공하는 핵심적인 부분입니다.^16^

### **프로액티브 트리거 아키텍처 비교**

두 패턴은 각각의 장단점과 적합한 사용 사례가 명확히 구분됩니다. 개발자는
구현하려는 기능의 요구사항을 분석하여 가장 적절한 아키텍처를 선택해야
합니다.

| 기능                   | 이벤트 기반 (API)                                                                                   | 시간 기반 (스케줄러)                                                                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **주요 사용 사례**     | 특정하고 예측 불가능한 이벤트에 대한 실시간 반응 (예: 사용자 가입, 구매, 지원 티켓 생성).           | 반복적이고 예측 가능한 배치 지향 작업 (예: 일일 보고서, 알림, 주기적인 확인).                                                        |
| **트리거 메커니즘**    | 외부 HTTP 요청 (예: 웹훅, 클라이언트 측 이벤트).                                                    | 내부적으로 사전 정의된 시간/간격 (예: cron, interval).                                                                               |
| **지연 시간(Latency)** | 낮음 (네트워크 및 모델 추론 시간에 의해 제한되는 거의 실시간).                                      | 높음 (스케줄에 의해 결정되는 내재적 지연).                                                                                           |
| **복잡성**             | 중간 \~ 높음. 견고한 API 인프라 필요: 보안(인증/인가), 속도 제한, 공용 엔드포인트 관리 및 확장.^12^ | 낮음 \~ 중간. 스케줄러가 항상 실행되도록 프로세스 관리 필요. 잡 지속성 및 다중 워커 설정은 복잡성을 증가시킬 수 있음.^16^            |
| **확장성**             | 높음. 표준 웹 확장 기술(로드 밸런서, 다중 인스턴스)을 사용하여 수평적으로 확장 가능.^12^            | 중간. 스케줄러를 실행하는 기계의 리소스에 따라 확장됨. 여러 워커에 분산하기 복잡할 수 있음.                                          |
| **데이터 소스**        | 트리거 이벤트 페이로드에 일반적으로 필요한 모든 컨텍스트가 포함됨.                                  | 잡 함수가 컨텍스트를 수집하기 위해 데이터베이스나 다른 소스를 쿼리해야 할 수 있음 (예: \"오늘 알림이 필요한 모든 사용자 가져오기\"). |

## **제 3부: 프로액티브 대화를 위한 그래프 내부 로직 설계**

프로액티브 트리거를 통해 그래프가 성공적으로 호출되었다면, 이제 그래프
내부에서 이 특별한 시작을 어떻게 처리하고 자연스러운 대화로 전환할
것인지 설계해야 합니다. 이 과정은 AI가 먼저 건네는 첫마디를 생성하는
\'Initiator\' 노드, 사용자의 응답을 기다리기 위해 실행을 일시 중지하는
\'Transition\' 로직, 그리고 이 모든 것을 아우르는 상태 관리로
구성됩니다.

### **3.1 \'Initiator\' 노드: 첫마디를 만드는 기술** {#initiator-노드-첫마디를-만드는-기술}

프로액티브 흐름에서 실행되는 첫 번째 노드인 \'Initiator\' 노드는 초기
상태에 담긴 trigger_context를 소비하여 대화의 첫 메시지, 즉 AIMessage를
생성하는 책임을 집니다. 이 첫 메시지의 품질이 사용자의 참여를 유도하는
데 결정적인 역할을 하므로, 맥락에 맞게 개인화되고 명확한 목적을 전달해야
합니다.^21^

Initiator 노드를 구현하는 전략은 크게 두 가지로 나눌 수 있습니다.

1.  **템플릿 기반(Template-Based) 생성:** \'user_signup\'이나
    > \'daily_briefing\'처럼 트리거 유형이 명확하고 메시지 구조가 비교적
    > 고정적인 경우, 미리 정의된 f-string이나 프롬프트 템플릿을 사용하는
    > 것이 효율적입니다. 이 방식은 빠르고 비용이 저렴하며 일관된
    > 결과물을 보장합니다.

2.  **LLM 기반(LLM-Powered) 생성:** \'post_purchase\'처럼 사용자가
    > 구매한 상품이나 과거 상호작용 기록 등 더 복잡한 맥락을 바탕으로
    > 역동적이고 고도로 개인화된 메시지를 생성해야 할 경우, LLM을
    > 호출하는 것이 효과적입니다. 이 때, trigger_context의 모든 정보를
    > 입력으로 받는 특화된 프롬프트를 사용하여 LLM이 창의적이고 매력적인
    > 첫인사를 만들도록 유도할 수 있습니다. 프롬프트 엔지니어링 기법을
    > 적극 활용하여 원하는 톤앤매너와 정보 포함 여부를 제어해야
    > 합니다.^23^

다음은 trigger_event_type에 따라 분기하여 첫 메시지를 생성하는 Initiator
노드의 Python 코드 예제입니다.

> Python

\# agent/nodes.py  
from langchain_core.messages import AIMessage, HumanMessage,
SystemMessage  
from.state import AgentState  
import logging  
  
\# 가상의 데이터베이스 조회 함수  
def \_fetch_user_profile_from_db(user_id: str) -\> dict:  
logging.info(f\"Fetching profile for user: {user_id}\")  
\# 실제로는 DB에서 사용자 정보를 조회합니다.  
profiles = {  
\"user_001\": {\"name\": \"Alice\", \"plan\": \"premium\"},  
\"user_002\": {\"name\": \"Bob\", \"plan\": \"free\"}  
}  
return profiles.get(user_id, {\"name\": \"Guest\", \"plan\":
\"unknown\"})  
  
def proactive_initiator_node(state: AgentState) -\> dict:  
\"\"\"  
trigger_context를 기반으로 프로액티브 대화의 첫 AI 메시지를
생성합니다.  
\"\"\"  
trigger_context = state.get(\"trigger_context\")  
if not trigger_context:  
\# 프로액티브 컨텍스트가 없으면 아무 작업도 하지 않고 다음 노드로 전달  
return {}  
  
user_id = trigger_context.get(\"user_id\")  
event_type = trigger_context.get(\"trigger_event_type\")  
  
\# 사용자 프로필 조회 및 상태 업데이트  
user_profile = \_fetch_user_profile_from_db(user_id)  
  
first_message_content = \"\"  
  
if event_type == \'user_signup\':  
user_name = user_profile.get(\"name\", \"고객\")  
first_message_content = f\"안녕하세요, {user_name}님! 저희 서비스에
가입해 주셔서 정말 기쁩니다. 혹시 저희 서비스를 어떻게 사용하면 좋을지
간단한 안내가 필요하신가요?\"  
  
elif event_type == \'post_purchase\':  
product_name = trigger_context.get(\"metadata\",
{}).get(\"product_name\", \"구매하신 상품\")  
first_message_content = f\"{user_profile.get(\'name\')}님, 최근에
구매하신 \'{product_name}\'은 마음에 드시나요? 제품 활용에 도움이 될
만한 팁을 알려드릴까요?\"  
  
elif event_type == \'daily_briefing\':  
\# 이 경우, LLM을 호출하여 개인화된 브리핑 생성  
\# llm = ChatOpenAI(\...)  
\# prompt = f\"{user_profile.get(\'name\')}님을 위한 오늘의 주요 뉴스
브리핑을 생성해 주세요.\"  
\# response = llm.invoke(prompt)  
\# first_message_content = response.content  
first_message_content = f\"좋은 아침입니다,
{user_profile.get(\'name\')}님! 오늘의 맞춤 브리핑을 전해드립니다. 가장
먼저 어떤 소식을 확인하고 싶으신가요?\"  
  
else:  
first_message_content = \"안녕하세요! 먼저 연락드렸습니다. 무엇을
도와드릴까요?\"  
  
\# 생성된 첫 메시지를 AIMessage로 만들어 상태에 추가  
initial_ai_message = AIMessage(content=first_message_content)  
  
\# user_profile도 상태에 추가하여 대화 전반에 걸쳐 활용  
return {  
\"messages\": \[initial_ai_message\],  
\"user_profile\": user_profile  
}

이 노드는 trigger_context를 분석하고, 필요한 경우 데이터베이스에서 추가
정보(사용자 프로필)를 조회한 뒤, 맥락에 맞는 AIMessage를 생성하여
그래프의 messages 상태를 업데이트합니다.^24^

### **3.2 전환점: 프로액티브 독백에서 반응형 대화로** {#전환점-프로액티브-독백에서-반응형-대화로}

AI가 첫 메시지를 성공적으로 보낸 후, 그래프는 이제 사용자의 응답을
기다려야 합니다. 만약 Initiator 노드에서 END로 바로 엣지를 연결하면,
대화는 AI의 일방적인 메시지 전송 후 즉시 종료되어 버립니다. 이는 우리가
원하는 양방향 상호작용이 아닙니다. 이 문제를 해결하고 프로액티브
독백(monologue)에서 반응형 대화(dialogue)로 자연스럽게 전환하기 위한
핵심 메커니즘이 바로 LangGraph의 interrupt 기능입니다.

interrupt는 langgraph.types에서 제공하는 특별한 객체로, 그래프의 실행을
그 자리에서 무기한 일시 중지시키는 역할을 합니다.^15^ 이 기능을 활용하여
\'사용자 응답 대기\' 노드를 만들 수 있습니다. 이 노드의 유일한 목적은

interrupt를 반환하여 그래프를 멈추고, 제어권을 외부 애플리케이션(사용자
인터페이스를 관리하는)으로 넘기는 것입니다.

> Python

\# agent/nodes.py (추가)  
from langgraph.types import interrupt  
  
def wait_for_user_response_node(state: AgentState) -\> dict:  
\"\"\"  
그래프 실행을 일시 중지하고 사용자 입력을 기다립니다.  
\"\"\"  
logging.info(\"Graph is interrupting, waiting for user input.\")  
\# interrupt 객체를 반환하면 해당 스레드의 실행이 여기서 멈춥니다.  
return interrupt()

이 wait_for_user_response_node를 Initiator 노드 다음에 연결하면, AI의 첫
메시지가 생성된 직후 그래프는 대기 상태에 들어갑니다.

이제 사용자가 응답할 차례입니다. 사용자가 채팅 인터페이스를 통해
메시지를 보내면, FastAPI 서버와 같은 외부 애플리케이션이 이 입력을
받습니다. 그 다음, 이전에 interrupt가 발생했던 동일한 thread_id를
사용하여 graph.ainvoke() 또는 관련 스트리밍 메서드를 다시 호출합니다.
이때, 새로운 입력은 사용자의 메시지를 포함한 상태 업데이트 형태로
전달됩니다. LangGraph의 checkpointer는 thread_id를 기반으로 중단되었던
상태를 정확히 복원하고, 새로운 사용자 메시지를 기존 대화 기록에 추가한
뒤, 중단된 지점부터 실행을 재개합니다.

이 interrupt 메커니즘은 **프로액티브 상태와 반응형 상태를 연결하는
다리** 역할을 합니다. 이는 시스템 주도(AI-driven)의 시작에서 사용자
주도(user-driven)의 대화로 제어권을 넘기는 과정을 명시적이고 공식적으로
처리하는 방법입니다. 임시적인 while 루프나 time.sleep과 같은 불안정한
방식 대신, interrupt는 LangGraph의 Human-in-the-Loop 기능을 활용하여
견고하고 디버깅하기 쉬운 전환점을 만들어냅니다.^27^

### **3.3 사용자 무반응 및 타임아웃 처리** {#사용자-무반응-및-타임아웃-처리}

만약 사용자가 AI의 프로액티브 메시지에 전혀 응답하지 않는다면 어떻게
될까요? interrupt 상태에 들어간 대화 스레드는 영원히 중단된 상태로 남게
되어 리소스를 낭비하고 상태 관리를 복잡하게 만듭니다. 이를
\'고스팅(ghosting)\' 문제라고 할 수 있습니다. LangGraph의 interrupt에는
내장된 타임아웃 기능이 없으므로, 이 문제를 해결하기 위해서는 아키텍처
수준의 접근이 필요합니다.

1.  **스케줄러 기반 타임아웃 처리:** 패턴 II에서 사용한 APScheduler를
    > 활용하여 주기적으로 실행되는 \'정리(cleanup)\' 잡을 만들 수
    > 있습니다. 이 잡은 checkpointer가 관리하는 영구 저장소(예:
    > 데이터베이스)를 쿼리하여, 특정 시간(예: 24시간) 이상 interrupt
    > 상태에 머물러 있는 스레드를 찾아냅니다. 그 후, 해당 스레드에 대해
    > \"timeout\"과 같은 특별한 메시지를 담아 graph.invoke()를 호출하여
    > 그래프가 정상적으로 END 상태로 전환되도록 할 수 있습니다. 이는
    > 서버 측에서 능동적으로 방치된 대화를 정리하는 가장 견고한
    > 방법입니다.

2.  **애플리케이션 로직 기반 타임아웃 처리:** 사용자가 애플리케이션을
    > 다시 방문했을 때, 새로운 대화를 시작하기 전에 마지막 메시지의
    > 타임스탬프를 확인하는 로직을 클라이언트나 API 서버에 구현할 수
    > 있습니다. 만약 마지막 메시지가 AI의 프로액티브 메시지였고, 일정
    > 시간이 지났다면 오래된 스레드를 재개하는 대신 새로운 대화 스레드를
    > 시작하도록 결정할 수 있습니다. 이 방식은 그래프 외부의
    > 애플리케이션 로직으로 타임아웃을 처리하는 더 간단한 접근법입니다.

어떤 방식을 선택하든, 장시간 응답이 없는 대화를 관리하는 정책을 수립하는
것은 프로덕션급 프로액티브 에이전트를 구축하는 데 있어 필수적인
고려사항입니다.

## **제 4부: 프로덕션급 운영을 위한 고려사항 및 모범 사례**

프로액티브 에이전트를 성공적으로 개발하는 것을 넘어, 실제 운영 환경에서
안정적이고 효과적으로 서비스하기 위해서는 기능적 요구사항 외에 여러
중요한 측면을 고려해야 합니다. 상태 관리의 지속성, 이벤트 처리의 멱등성,
개인화, 보안 및 개인정보 보호, 그리고 견고한 오류 처리 전략은 프로덕션급
시스템의 품질을 결정하는 핵심 요소입니다.

### **4.1 상태 지속성 및 멱등성** {#상태-지속성-및-멱등성}

장시간 실행되거나 사용자의 응답을 기다려야 하는 프로액티브 에이전트에게
**Checkpointer**는 선택이 아닌 필수입니다. Checkpointer는 그래프의
상태를 외부 저장소(예: 인메모리, 파일, Redis, PostgreSQL)에 지속적으로
저장하여, 애플리케이션이 예기치 않게 재시작되더라도 대화의 맥락이
손실되지 않도록 보장합니다.^26^ 개발 중에는

MemorySaver를 사용할 수 있지만, 프로덕션 환경에서는 반드시 Redis나
데이터베이스 기반의 영구적인 checkpointer를 사용해야 합니다.

또한, 프로액티브 트리거, 특히 웹훅을 통해 전달되는 이벤트는 네트워크
문제나 재시도 로직으로 인해 동일한 이벤트가 여러 번 전송될 수 있습니다.
만약 시스템이 이를 제대로 처리하지 못하면, 봇은 사용자에게 동일한 환영
메시지나 구매 감사 메시지를 여러 번 보내는 치명적인 실수를 저지를 수
있습니다. 이러한 중복 실행을 방지하기 위해 시스템은
\*\*멱등성(Idempotency)\*\*을 보장해야 합니다.

멱등성을 구현하는 가장 효과적인 방법은 **조건부 진입점과 이벤트 ID를
활용**하는 것입니다. 프로액티브 상태 스키마에 trigger_event_id라는 고유
식별자를 포함시키고, 영구 checkpointer를 사용함으로써 이 문제를 해결할
수 있습니다. 그래프의 첫 번째 노드는 단순히 메시지를 생성하는 대신,
조건부 라우터(conditional router) 역할을 하도록 설계합니다. 이 라우터의
로직은 다음과 같습니다.

1.  현재 thread_id에 대한 전체 상태 기록을 checkpointer를 통해
    > 가져옵니다.

2.  가져온 상태 기록 전체를 스캔하여, 현재 요청으로 들어온
    > trigger_event_id와 동일한 ID를 가진 trigger_context가 이미
    > 존재하는지 확인합니다.

3.  만약 존재한다면, 이는 중복된 요청이므로 즉시 END 노드로 라우팅하여
    > 아무런 추가 작업을 수행하지 않고 그래프를 종료합니다.

4.  만약 존재하지 않는다면, 이는 새로운 이벤트이므로 정상적인
    > proactive_initiator 노드로 라우팅하여 대화를 시작합니다.

이 패턴은 LangGraph의 핵심 기능인 상태 지속성을 활용하여 외부 시스템의
불안정성으로부터 내부 로직을 보호하는 견고하고 신뢰할 수 있는 시스템을
구축하는 방법을 보여줍니다. 이는 프로덕션급 API를 설계할 때 반드시
고려해야 할 중요한 아키텍처 원칙입니다.

### **4.2 개인화의 중요성** {#개인화의-중요성}

프로액티브 메시지가 성공하기 위한 가장 중요한 조건은
\*\*개인화(Personalization)\*\*입니다. 맥락 없이 무차별적으로 전송되는
메시지는 사용자에게 환영받기보다는 스팸으로 인식될 가능성이 높습니다.
연구에 따르면, 사용자들은 개인화된 경험을 기대하며, 이러한 경험이 브랜드
선택에 큰 영향을 미친다고 응답했습니다.^21^ 82%의 소비자가 개인화된
경험이 브랜드 선택에 영향을 미친다고 믿으며, 71%는 개인화된 상호작용을
기대합니다.^21^

우리가 설계한 AgentState의 trigger_context와 user_profile 필드는
개인화를 구현하기 위한 핵심적인 열쇠입니다. Initiator 노드는 이 데이터를
적극적으로 활용해야 합니다. 예를 들어, user_id를 사용하여
데이터베이스에서 사용자의 이름, 과거 구매 내역, 서비스 이용 패턴 등을
조회하고, trigger_context의 metadata에 포함된 상품 ID나 이벤트 정보를
결합하여 \"Alice님, 지난주에 구매하신 \'스마트 워치\'와 함께 사용하면
좋은 스트랩 신제품이 출시되었는데, 확인해 보시겠어요?\"와 같이 매우
구체적이고 사용자에게 실질적인 가치를 제공하는 메시지를 생성할 수
있습니다.^30^ 이러한 고도의 개인화는 사용자의 참여를 유도하고 긍정적인
관계를 형성하는 데 결정적인 역할을 합니다.

### **4.3 보안, 개인정보, 그리고 사용자 동의** {#보안-개인정보-그리고-사용자-동의}

사용자에게 먼저 연락을 취하는 행위는 상당한 책임감을 수반합니다. 특히
개인정보를 다루는 경우, 법적 및 윤리적 의무를 준수하는 것이 매우
중요합니다. 유럽의 GDPR(일반 데이터 보호 규정)과 같은 규제는 데이터 처리
및 통신에 대한 명시적인 사용자 동의를 요구합니다.^21^

프로액티브 에이전트 설계 시 다음의 모범 사례를 반드시 준수해야 합니다.

- **명시적 동의(Opt-in):** 사용자가 회원 가입 과정이나 프로필 설정에서
  > 프로액티브 커뮤니케이션 수신에 명시적으로 동의했는지 확인해야
  > 합니다.

- **쉬운 거부(Opt-out):** 사용자가 향후 프로액티브 메시지 수신을 원하지
  > 않을 경우, 이를 쉽고 명확하게 거부할 수 있는 방법을 제공해야 합니다.
  > 이는 에이전트가 호출할 수 있는 도구(tool) 형태로 구현될 수 있습니다.

- **투명성:** AI가 왜 연락했는지 명확하게 밝혀야 합니다. \"최근 구매하신
  > 상품에 대한 만족도 조사를 위해 연락드렸습니다.\"와 같이 목적을
  > 투명하게 공개하는 것이 신뢰를 구축하는 데 도움이 됩니다.

- **데이터 보안:** 모든 사용자 데이터, 특히 개인 식별 정보(PII)는 전송
  > 중(in-transit) 및 저장 시(at-rest) 암호화하여 안전하게 보호해야
  > 합니다. SSL/TLS 및 AES와 같은 표준 암호화 기술을 적용하는 것이
  > 필수적입니다.^31^

### **4.4 견고성 및 오류 처리** {#견고성-및-오류-처리}

안정적인 서비스 운영을 위해서는 시스템의 모든 계층에서 발생할 수 있는
오류를 예측하고 처리하는 전략이 필요합니다.

- **외부 트리거 오류:** API 엔드포인트는 잘못된 요청에 대해 400번대 HTTP
  > 상태 코드를, 서버 내부 오류에 대해서는 500번대 코드를 반환하는 등
  > 표준적인 오류 처리를 구현해야 합니다. 스케줄러는 잡 실행 실패 시
  > 이를 감지하고 로깅할 수 있는 포괄적인 로깅 시스템을 갖추어야
  > 합니다.^32^

- **그래프 내부 오류:** LangGraph의 각 노드는 try\...except 블록으로
  > 감싸서 외부 API 호출 실패나 데이터 처리 오류와 같은 예외 상황을
  > 처리해야 합니다. 예외가 발생했을 때 그래프 전체가 중단되는 대신,
  > 오류 정보를 로깅하고 사용자에게 \"시스템에 일시적인 문제가
  > 발생했습니다. 잠시 후 다시 시도해 주세요.\"와 같은 안내 메시지를
  > 전달하는 전용 \'오류 처리\' 노드로 흐름을 유도하여
  > 우아하게(gracefully) 상황을 마무리하는 것이 좋습니다.^2^

## **결론: 앰비언트 인텔리전스로의 길**

본 보고서는 LangGraph를 사용하여 사용자의 입력에만 의존하는 수동적인
챗봇을 넘어, 비즈니스 목표와 사용자 맥락에 따라 먼저 대화를 시작하는
능동적인 프로액티브 에이전트를 구축하는 포괄적인 방법론을 제시했습니다.
핵심적인 아키텍처적 통찰은 다음과 같이 요약할 수 있습니다.

첫째, **트리거와 로직의 분리**입니다. 프로액티브 기능은 LangGraph의 특정
함수가 아닌, API나 스케줄러와 같은 외부 시스템이 그래프를 \'호출\'하는
아키텍처적 선택의 문제입니다. 이 원칙은 에이전트의 재사용성과 모듈성을
극대화합니다.

둘째, **초기 상태의 역할**입니다. invoke 함수에 전달되는 초기 상태
객체는 단순한 입력이 아니라, trigger_context를 통해 전체 대화의 경로를
결정하는 핵심적인 제어 메커니즘으로 기능합니다.

셋째, **interrupt를 통한 전환**입니다. LangGraph의 interrupt 기능은 AI가
주도하는 프로액티브 시작에서 사용자가 주도하는 반응형 대화로 제어권을
자연스럽게 넘기는, 견고하고 명시적인 \'다리\' 역할을 수행합니다.

마지막으로, **상태 지속성을 통한 멱등성 확보**입니다. Checkpointer와
고유 이벤트 ID를 결합하여, 외부 트리거의 중복 호출에도 시스템이
안정적으로 동작하도록 보장하는 것은 프로덕션 환경의 필수 요건입니다.

이러한 프로액티브 패턴들은 단순히 AI가 먼저 말을 거는 기능을 구현하는
것을 넘어, 사용자의 필요를 예측하고 자율적으로 행동하며, 그들의 디지털
생활에 더 깊고 유용하게 통합되는 차세대 \'앰비언트 에이전트(Ambient
Agents)\'로 나아가는 중요한 첫걸음입니다.^29^ 개발자들은 본 보고서에서
제시된 아키텍처 원칙과 구현 패턴을 기반으로, 더욱 지능적이고 사용자
중심적인 AI 서비스를 구축할 수 있을 것입니다.

#### 참고 자료

1.  LangGraph - LangChain Blog, 7월 13, 2025에 액세스,
    > [[https://blog.langchain.dev/langgraph/]{.underline}](https://blog.langchain.dev/langgraph/)

2.  LangGraph Simplified - Kaggle, 7월 13, 2025에 액세스,
    > [[https://www.kaggle.com/code/marcinrutecki/langgraph-simplified]{.underline}](https://www.kaggle.com/code/marcinrutecki/langgraph-simplified)

3.  LangGraph Tutorial: What Is LangGraph and How to Use It? - DataCamp,
    > 7월 13, 2025에 액세스,
    > [[https://www.datacamp.com/tutorial/langgraph-tutorial]{.underline}](https://www.datacamp.com/tutorial/langgraph-tutorial)

4.  1\. Build a basic chatbot - GitHub Pages, 7월 13, 2025에 액세스,
    > [[https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/]{.underline}](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)

5.  struggling to understand the langgraph tutorial (build a basic
    > chatbot) : r/LangChain - Reddit, 7월 13, 2025에 액세스,
    > [[https://www.reddit.com/r/LangChain/comments/1heg7pi/struggling_to_understand_the_langgraph_tutorial/]{.underline}](https://www.reddit.com/r/LangChain/comments/1heg7pi/struggling_to_understand_the_langgraph_tutorial/)

6.  Build a Chatbot \| 🦜️ LangChain, 7월 13, 2025에 액세스,
    > [[https://python.langchain.com/docs/tutorials/chatbot/]{.underline}](https://python.langchain.com/docs/tutorials/chatbot/)

7.  How to use BaseChatMessageHistory with LangGraph \| 🦜️ LangChain,
    > 7월 13, 2025에 액세스,
    > [[https://python.langchain.com/docs/versions/migrating_memory/chat_history/]{.underline}](https://python.langchain.com/docs/versions/migrating_memory/chat_history/)

8.  LangGraph Tutorial: Building an Advanced Stateful Conversation
    > System - Unit 1.1 Exercise 5 - AI Product Engineer, 7월 13, 2025에
    > 액세스,
    > [[https://aiproduct.engineer/tutorials/langgraph-tutorial-building-an-advanced-stateful-conversation-system-unit-11-exercise-5]{.underline}](https://aiproduct.engineer/tutorials/langgraph-tutorial-building-an-advanced-stateful-conversation-system-unit-11-exercise-5)

9.  Understanding State in LangGraph: A Beginners Guide \| by Rick
    > Garcia \| Medium, 7월 13, 2025에 액세스,
    > [[https://medium.com/@gitmaxd/understanding-state-in-langgraph-a-comprehensive-guide-191462220997]{.underline}](https://medium.com/@gitmaxd/understanding-state-in-langgraph-a-comprehensive-guide-191462220997)

10. How do I change the name of the langgraph in langsmith \#22182 -
    > GitHub, 7월 13, 2025에 액세스,
    > [[https://github.com/langchain-ai/langchain/discussions/22182]{.underline}](https://github.com/langchain-ai/langchain/discussions/22182)

11. Create a RAG Chatbot with LangGraph and FastAPI: A Step-by-Step
    > Guide - Medium, 7월 13, 2025에 액세스,
    > [[https://medium.com/codex/create-a-rag-chatbot-with-langgraph-and-fastapi-a-step-by-step-guide-4c2fbc33ed46]{.underline}](https://medium.com/codex/create-a-rag-chatbot-with-langgraph-and-fastapi-a-step-by-step-guide-4c2fbc33ed46)

12. wassim249/fastapi-langgraph-agent-production-ready-template -
    > GitHub, 7월 13, 2025에 액세스,
    > [[https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template]{.underline}](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template)

13. How to use LangGraph within a FastAPI Backend - DEV Community, 7월
    > 13, 2025에 액세스,
    > [[https://dev.to/anuragkanojiya/how-to-use-langgraph-within-a-fastapi-backend-amm]{.underline}](https://dev.to/anuragkanojiya/how-to-use-langgraph-within-a-fastapi-backend-amm)

14. Deploying LangGraph with FastAPI: A Step-by-Step Tutorial \| by
    > Sajith K - Medium, 7월 13, 2025에 액세스,
    > [[https://medium.com/@sajith_k/deploying-langgraph-with-fastapi-a-step-by-step-tutorial-b5b7cdc91385]{.underline}](https://medium.com/@sajith_k/deploying-langgraph-with-fastapi-a-step-by-step-tutorial-b5b7cdc91385)

15. langgraph/concepts/human_in_the_loop/ \#2290 - GitHub, 7월 13,
    > 2025에 액세스,
    > [[https://github.com/langchain-ai/langgraph/discussions/2290]{.underline}](https://github.com/langchain-ai/langgraph/discussions/2290)

16. Job Scheduling in Python with APScheduler \| Better Stack Community,
    > 7월 13, 2025에 액세스,
    > [[https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/]{.underline}](https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/)

17. User guide --- APScheduler 3.11.0.post1 documentation - Read the
    > Docs, 7월 13, 2025에 액세스,
    > [[https://apscheduler.readthedocs.io/en/3.x/userguide.html]{.underline}](https://apscheduler.readthedocs.io/en/3.x/userguide.html)

18. Scheduled Tasks in LangGraph - YouTube, 7월 13, 2025에 액세스,
    > [[https://www.youtube.com/watch?v=9DRn9RpR2vA]{.underline}](https://www.youtube.com/watch?v=9DRn9RpR2vA)

19. Cron jobs - Overview, 7월 13, 2025에 액세스,
    > [[https://langchain-ai.github.io/langgraph/cloud/concepts/cron_jobs/]{.underline}](https://langchain-ai.github.io/langgraph/cloud/concepts/cron_jobs/)

20. Hardening the RAG chatbot architecture powered by Amazon Bedrock:
    > Blueprint for secure design and anti-pattern mitigation - AWS, 7월
    > 13, 2025에 액세스,
    > [[https://aws.amazon.com/blogs/security/hardening-the-rag-chatbot-architecture-powered-by-amazon-bedrock-blueprint-for-secure-design-and-anti-pattern-migration/]{.underline}](https://aws.amazon.com/blogs/security/hardening-the-rag-chatbot-architecture-powered-by-amazon-bedrock-blueprint-for-secure-design-and-anti-pattern-migration/)

21. 9 Principles for Effective Chatbot Design in Business - agentics,
    > 7월 13, 2025에 액세스,
    > [[https://agentics.uk/user-centric-ai-design/9-principles-for-effective-chatbot-design-in-business/]{.underline}](https://agentics.uk/user-centric-ai-design/9-principles-for-effective-chatbot-design-in-business/)

22. Chatbot Design Guidelines - SmythOS, 7월 13, 2025에 액세스,
    > [[https://smythos.com/ai-agents/chatbots/chatbot-design-guidelines/]{.underline}](https://smythos.com/ai-agents/chatbots/chatbot-design-guidelines/)

23. LangChain & LangGraph Tutorial: In-Depth Chat Memory & Prompts -
    > Triumph.ai, 7월 13, 2025에 액세스,
    > [[https://www.triumphai.in/post/learn-langchain-langgraph-in-depth-chat-memory-prompts]{.underline}](https://www.triumphai.in/post/learn-langchain-langgraph-in-depth-chat-memory-prompts)

24. Use the Graph API - GitHub Pages, 7월 13, 2025에 액세스,
    > [[https://langchain-ai.github.io/langgraph/how-tos/graph-api/]{.underline}](https://langchain-ai.github.io/langgraph/how-tos/graph-api/)

25. Built with LangGraph! \#6: First Node \| by Okan Yenigün \| Jul,
    > 2025 \| Medium, 7월 13, 2025에 액세스,
    > [[https://medium.com/@okanyenigun/built-with-langgraph-6-first-node-d61fdc30bca4]{.underline}](https://medium.com/@okanyenigun/built-with-langgraph-6-first-node-d61fdc30bca4)

26. Introducing the LangGraph Functional API - LangChain Blog, 7월 13,
    > 2025에 액세스,
    > [[https://blog.langchain.com/introducing-the-langgraph-functional-api/]{.underline}](https://blog.langchain.com/introducing-the-langgraph-functional-api/)

27. Learn LangGraph basics - Overview, 7월 13, 2025에 액세스,
    > [[https://langchain-ai.github.io/langgraph/concepts/why-langgraph/]{.underline}](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)

28. LangGraph - LangChain, 7월 13, 2025에 액세스,
    > [[https://www.langchain.com/langgraph]{.underline}](https://www.langchain.com/langgraph)

29. Build a Customer Support Bot - GitHub Pages, 7월 13, 2025에 액세스,
    > [[https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/]{.underline}](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/)

30. Design principles and features of the best chatbot - RoboticsBiz,
    > 7월 13, 2025에 액세스,
    > [[https://roboticsbiz.com/design-principles-and-features-of-the-best-chatbot/]{.underline}](https://roboticsbiz.com/design-principles-and-features-of-the-best-chatbot/)

31. Chatbot Architecture Design: Key Principles for Building Intelligent
    > Bots - FastBots.ai, 7월 13, 2025에 액세스,
    > [[https://fastbots.ai/blog/chatbot-architecture-design-key-principles-for-building-intelligent-bots]{.underline}](https://fastbots.ai/blog/chatbot-architecture-design-key-principles-for-building-intelligent-bots)

32. Automate AI Workflows with Cron Jobs in LangGraph: Daily Summaries
    > Example - Medium, 7월 13, 2025에 액세스,
    > [[https://medium.com/@sangeethasaravanan/automate-ai-workflows-with-cron-jobs-in-langgraph-daily-summaries-example-be2908a4c615]{.underline}](https://medium.com/@sangeethasaravanan/automate-ai-workflows-with-cron-jobs-in-langgraph-daily-summaries-example-be2908a4c615)
