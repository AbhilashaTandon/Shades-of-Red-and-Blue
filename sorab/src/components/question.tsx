import { ReactElement } from "react";
import Styles from "@/styles/question.module.css";

export default function Question({ question, options }: { question: string; options: string[] }) {
  const options_keys: { answer: string; id: number }[] = [];
  options.forEach((answer: string, index: number) => {
    options_keys.push({ answer: answer, id: index });
  });

  return (
    <div className={Styles.question}>
      <h5>{question}</h5>
      <div className={Styles.answers}>
        {options_keys.map((option) => (
          <div
            style={{ display: "inline", float: "left" }}
            key={option.id}>
            <p> {option.answer} </p>
            <input
              type="radio"
              id={option.answer}
              value={option.answer}></input>
          </div>
        ))}
      </div>
    </div>
  );
}
