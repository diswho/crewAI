from asyncio import Task
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent
from langchain_groq import ChatGroq
import os
from crewai import Crew, Process

agent_finishes = []


call_number = 0

search_tool = DuckDuckGoSearchRun()
search_tool.run("usb key price")

GROQ_LLM = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)


class ResearchAgents():
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
            tools=[search_tool],
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
            tools=[search_tool],
            allow_delegation=False
        )

    def make_manager(self):
        return Agent(
            role='Manager',
            goal='Ensure the smooth operation and coordination of the team',
            verbose=True,
            backstory=(
                "As a seasoned project manager, you excel in organizing"
                "tasks, managing timelines, and ensuring the team stays on track."
            )
        )


class ResearchTask():
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


agents = ResearchAgents()
task = ResearchTask()

researcher = agents.make_researcher()
writer = agents.make_writer()
manager = agents.make_manager()

crew = Crew(
    agents=[researcher, writer],
    tasks=[search_tool],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    manager_agent=manager
)
result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
print(result)
