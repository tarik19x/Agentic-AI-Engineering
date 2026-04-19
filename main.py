from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main(): 
    print("RUNNING NEW CODE")
    information ="""
    Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman, inventor,[2] and investor. A pioneer of the personal computer revolution of the 1970s and 1980s, Jobs co-founded Apple Inc. with his early business partner Steve Wozniak as Apple Computer Company in 1976. After the company's board of directors fired him in 1985, he founded NeXT the same year and purchased Pixar in 1986, becoming its chairman and majority shareholder until 2007. Jobs returned to Apple in 1997 as CEO, where he was closely involved with the creation and promotion of many of the company's most influential products until his resignation in 2011.

    Jobs was born in San Francisco in 1955 and adopted shortly afterward. He attended Reed College in 1972 before withdrawing that same year. In 1974, he traveled through India, seeking enlightenment before later studying Zen Buddhism. He and Wozniak co-founded Apple in 1976 to further develop and sell Wozniak's Apple I personal computer. Together, the duo gained fame and wealth a year later with the production and sale of the Apple II, one of the first highly successful mass-produced microcomputers.

    Jobs saw the commercial potential of the Xerox Alto in 1979, which was mouse-driven and had a graphical user interface (GUI). This led to the development of the largely unsuccessful Apple Lisa in 1983, followed by the breakthrough Macintosh 128K in 1984, the first mass-produced computer with a GUI. The Macintosh launched the desktop publishing industry in 1985 (for example, the Aldus PageMaker) with the addition of the Apple LaserWriter, the first laser printer to feature vector graphics and PostScript.

    In 1985, Jobs departed Apple after a long power struggle with the company's board and its then-CEO, John Sculley. That same year, Jobs took some Apple employees with him to found NeXT, a computer platform development company that specialized in computers for higher education and business markets, serving as its CEO. In 1986, he bought the computer graphics division of Lucasfilm, which was spun off independently as Pixar.[3] Pixar produced the first computer-animated feature film, Toy Story (1995), and became a leading animation studio, producing dozens of commercially successful and critically acclaimed films.

    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_tempalte = PromptTemplate(input_variables = ["information"], template = summary_template)
    llm = ChatOpenAI(temperature= 0, model ="gpt-5")
    
    chain  = summary_prompt_tempalte | llm
    response = chain.invoke(information = information)

if __name__ == "__main__":
    main()
