import os

from pydantic import BaseModel
from crewai import Agent, Crew, LLM, Process, Task
from crewai.flow.flow import Flow, listen, router, start


# ---------------------------------------------------------
# 1. Flow State
# ---------------------------------------------------------

class ContentState(BaseModel):
    topic: str = ""
    research: str = ""
    draft: str = ""
    human_feedback: str = ""
    approval_status: str = ""
    final_content: str = ""
    revision_count: int = 0


# ---------------------------------------------------------
# 2. Local Ollama LLM
# ---------------------------------------------------------

llm = LLM(
    model="ollama/gpt-oss:latest",
    base_url="http://localhost:11434",
    temperature=0.2,
)


# ---------------------------------------------------------
# 3. CrewAI Flow
# ---------------------------------------------------------

class ContentFlow(Flow[ContentState]):

    # -----------------------------------------------------
    # STEP 1
    # -----------------------------------------------------

    @start()
    def get_topic(self):

        self.state.topic = "The future of Generative AI"

        print("\n================================")
        print("STEP 1: Topic")
        print("================================")

        print(f"Topic: {self.state.topic}")

        return self.state.topic


    # -----------------------------------------------------
    # STEP 2
    # -----------------------------------------------------

    @listen(get_topic)
    def research_content(self, topic):

        print("\n================================")
        print("STEP 2: Research Crew")
        print("================================")

        researcher = Agent(
            role="AI Researcher",
            goal=(
                "Research the given topic and identify "
                "important facts, trends and insights."
            ),
            backstory=(
                "You are an experienced technology researcher "
                "specializing in Generative AI."
            ),
            llm=llm,
            verbose=True,
        )

        writer = Agent(
            role="Technical Writer",
            goal=(
                "Create a clear and technically accurate "
                "article using the research."
            ),
            backstory=(
                "You are an experienced technical writer "
                "who explains complex AI topics clearly."
            ),
            llm=llm,
            verbose=True,
        )

        research_task = Task(
            description=f"""
            Research the following topic:

            {topic}

            Cover:
            - What it is
            - Current trends
            - Major use cases
            - Benefits
            - Challenges
            - Future possibilities

            Provide structured research notes.
            """,
            expected_output="Detailed research notes in Markdown.",
            agent=researcher,
        )

        writing_task = Task(
            description=f"""
            Create a technical article about:

            {topic}

            Use the research produced by the researcher.

            The article should contain:
            - Introduction
            - Key concepts
            - Use cases
            - Benefits
            - Challenges
            - Future outlook
            """,
            expected_output="A well-structured technical article in Markdown.",
            agent=writer,
            context=[research_task],
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        self.state.research = str(result)
        self.state.draft = str(result)

        print("\nDraft generated.")

        return self.state.draft


    # -----------------------------------------------------
    # STEP 3
    # -----------------------------------------------------

    @listen(research_content)
    def human_review(self, draft):

        print("\n================================")
        print("STEP 3: HUMAN REVIEW")
        print("================================")

        print("\n------------- DRAFT -------------")
        print(draft)
        print("\n----------------------------------")

        print("\nWhat do you want to do?")

        print("1. approve")
        print("2. edit")
        print("3. reject")

        choice = input("\nYour choice: ").strip().lower()

        self.state.approval_status = choice

        if choice == "edit":

            feedback = input(
                "\nEnter your feedback for the AI: "
            )

            self.state.human_feedback = feedback

        return choice


    # -----------------------------------------------------
    # STEP 4
    # -----------------------------------------------------

    @router(human_review)
    def route_human_decision(self, decision):

        print("\n================================")
        print("STEP 4: ROUTING")
        print("================================")

        if decision == "approve":

            print("Human decision → APPROVED")

            return "approved"

        elif decision == "edit":

            print("Human decision → EDIT")

            return "edit"

        elif decision == "reject":

            print("Human decision → REJECTED")

            return "rejected"

        else:

            print("Invalid decision")

            return "rejected"


    # -----------------------------------------------------
    # STEP 5A
    # -----------------------------------------------------

    @listen("approved")
    def publish_content(self):

        print("\n================================")
        print("STEP 5: PUBLISH")
        print("================================")

        self.state.final_content = self.state.draft

        print("Content approved by human.")
        print("Publishing content...")

        return self.state.final_content


    # -----------------------------------------------------
    # STEP 5B
    # -----------------------------------------------------

    @listen("edit")
    def revise_content(self):

        print("\n================================")
        print("STEP 5: AI REVISION")
        print("================================")

        self.state.revision_count += 1

        reviser = Agent(
            role="Content Editor",
            goal=(
                "Improve the article according to human feedback."
            ),
            backstory=(
                "You are a senior technical editor "
                "who improves AI-generated content."
            ),
            llm=llm,
            verbose=True,
        )

        revision_task = Task(
            description=f"""
            Improve the following article.

            ARTICLE:
            {self.state.draft}

            HUMAN FEEDBACK:
            {self.state.human_feedback}

            Apply the human's feedback carefully.

            Return the complete revised article.
            """,
            expected_output="Complete revised article in Markdown.",
            agent=reviser,
        )

        revision_crew = Crew(
            agents=[reviser],
            tasks=[revision_task],
            process=Process.sequential,
            verbose=True,
        )

        result = revision_crew.kickoff()

        self.state.draft = str(result)

        print("\nRevised draft:")
        print(self.state.draft)

        return self.state.draft


    # -----------------------------------------------------
    # STEP 6
    # -----------------------------------------------------

    @listen(revise_content)
    def second_human_review(self, draft):

        print("\n================================")
        print("STEP 6: SECOND HUMAN REVIEW")
        print("================================")

        print("\n------------- REVISED DRAFT -------------")
        print(draft)
        print("\n------------------------------------------")

        choice = input(
            "\nApprove revised content? (yes/no): "
        ).strip().lower()

        if choice == "yes":
            return "approved"

        return "rejected"


    # -----------------------------------------------------
    # STEP 7
    # -----------------------------------------------------

    @listen("rejected")
    def reject_content(self):

        print("\n================================")
        print("CONTENT REJECTED")
        print("================================")

        self.state.final_content = ""

        return "Content rejected by human."


# ---------------------------------------------------------
# 4. Run Flow
# ---------------------------------------------------------

def kickoff():
    flow = ContentFlow()

    result = flow.kickoff()

    print("\n================================")
    print("FLOW COMPLETED")
    print("================================")

    print(result)


if __name__ == "__main__":
    kickoff()