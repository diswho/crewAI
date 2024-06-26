import json
# from typing import Dict, List, Tuple, Union
from crewai import Crew, Process, Task, Agent
from langchain_community.tools import DuckDuckGoSearchRun
import os
# from langchain_core.agents import AgentFinish
from langchain_groq import ChatGroq

# agent_finishes = []


# call_number = 0


# def print_agent_output(agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish], agent_name: str = 'Generic call'):
#     global call_number  # Declare call_number as a global variable
#     call_number += 1
#     with open("crew_callback_logs.txt", "a") as log_file:
#         # Try to parse the output if it is a JSON string
#         if isinstance(agent_output, str):
#             try:
#                 agent_output = json.loads(agent_output)  # Attempt to parse the JSON string
#             except json.JSONDecodeError:
#                 pass  # If there's an error, leave agent_output as is

#         # Check if the output is a list of tuples as in the first case
#         if isinstance(agent_output, list) and all(isinstance(item, tuple) for item in agent_output):
#             print(f"-{call_number}----Dict------------------------------------------", file=log_file)
#             for action, description in agent_output:
#                 # Print attributes based on assumed structure
#                 print(f"Agent Name: {agent_name}", file=log_file)
#                 print(f"Tool used: {getattr(action, 'tool', 'Unknown')}", file=log_file)
#                 print(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}", file=log_file)
#                 print(f"Action log: {getattr(action, 'log', 'Unknown')}", file=log_file)
#                 print(f"Description: {description}", file=log_file)
#                 print("--------------------------------------------------", file=log_file)

#         # Check if the output is a dictionary as in the second case
#         elif isinstance(agent_output, AgentFinish):
#             print(f"-{call_number}----AgentFinish---------------------------------------", file=log_file)
#             print(f"Agent Name: {agent_name}", file=log_file)
#             agent_finishes.append(agent_output)
#             # Extracting 'output' and 'log' from the nested 'return_values' if they exist
#             output = agent_output.return_values
#             # log = agent_output.get('log', 'No log available')
#             print(f"AgentFinish Output: {output['output']}", file=log_file)
#             # print(f"Log: {log}", file=log_file)
#             # print(f"AgentFinish: {agent_output}", file=log_file)
#             print("--------------------------------------------------", file=log_file)

#         # Handle unexpected formats
#         else:
#             # If the format is unknown, print out the input directly
#             print(f"-{call_number}-Unknown format of agent_output:", file=log_file)
#             print(type(agent_output), file=log_file)
#             print(agent_output, file=log_file)


GROQ_LLM = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

search_tool = DuckDuckGoSearchRun()
search_tool.run("usb key price")


class ResearchAgent():
    # Creating a senior researcher agent with memory and verbose mode
    def make_researcher(self):
        return Agent(
            role='Senior Researcher',
            goal='Uncover groundbreaking technologies in {topic}',
            verbose=True,
            memory=True,
            backstory=(
                "Driven by curiosity, you're at the forefront of"
                "innovation, eager to explore and share knowledge that could change"
                "the world."
            ),
            llm=GROQ_LLM,
            tools=[search_tool, ],
        )
# Creating a writer agent with custom tools and delegation capability

    def make_writer(self):
        return Agent(
            role='Writer',
            goal='Narrate compelling tech stories about {topic}',
            verbose=True,
            memory=True,
            backstory=(
                "With a flair for simplifying complex topics, you craft"
                "engaging narratives that captivate and educate, bringing new"
                "discoveries to light in an accessible manner."
            ),
            llm=GROQ_LLM,
            tools=[search_tool],
            allow_delegation=False
        )
# Setting a specific manager agent

    def make_manager(self):
        return Agent(
            role='Manager',
            goal='Ensure the smooth operation and coordination of the team',
            verbose=True,
            backstory=(
                "As a seasoned project manager, you excel in organizing"
                "tasks, managing timelines, and ensuring the team stays on track."
            ),
            llm=GROQ_LLM,
        )


class ResearchTask():
    # Research task
    def research_task(self):
        return Task(
            description=(
                "Identify the next big trend in {topic}."
                "Focus on identifying pros and cons and the overall narrative."
                "Your final report should clearly articulate the key points,"
                "its market opportunities, and potential risks."
            ),
            expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
            tools=[search_tool],
            agent=researcher,
            callback="research_callback",  # Example of task callback
            human_input=True
        )


# Writing task with language model configuration


    def write_task(self):
        return Task(
            description=(
                "Compose an insightful article on {topic}."
                "Focus on the latest trends and how it's impacting the industry."
                "This article should be easy to understand, engaging, and positive."
            ),
            expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
            tools=[search_tool],
            agent=writer,
            output_file='new-blog-post.md',  # Example of output customization
        )


agents = ResearchAgent()
tasks = ResearchTask()
researcher = agents.make_researcher()
writer = agents.make_writer()
research_task = tasks.research_task()
write_task = tasks.write_task()

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    manager_agent=agents.make_manager
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
print(result)
