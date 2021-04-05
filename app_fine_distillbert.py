from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio
from flask import Flask
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

app = Flask(__name__)

loop = asyncio.get_event_loop()
bot = Bot(token="1709244299:AAEMfyW-nXkHm3C2x8aVIMBpqdeNYintcrg")
dp = Dispatcher(bot, loop)

@dp.message_handler(commands= ['start'])
async def main(message:types.Message):
    await message.reply('Welcome to QnA Bert Bot, this will work for qna from the below text, it wont work for hi , how are you etc')
    await message.reply(""" Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its share and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet leading subsidiary and will continue to be the umbrella company for Alphabets Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.).""")
   
@dp.message_handler()
async def main(message: types.Message):
    context = """Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its share and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet leading subsidiary and will continue to be the umbrella company for Alphabets Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.).."""

    modelname = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertForQuestionAnswering.from_pretrained(modelname)
    mid = message.message_id
    textin = message.text
    print("textin : ", textin)

    while True:
        print('bot received a question:', textin)
        if textin == "bye" or textin == "Bye":
            reply = "Bye, see you later."
            await message.answer(reply)
            break
        else:
            input_ids = tokenizer.encode(textin, context)
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # Search the input_ids for the first instance of the `[SEP]` token.
            sep_index = input_ids.index(tokenizer.sep_token_id)

            # The number of segment A tokens includes the [SEP] token istelf.
            num_seg_a = sep_index + 1

            # The remainder are segment B.
            num_seg_b = len(input_ids) - num_seg_a

            # Construct the list of 0s and 1s.
            segment_ids = [0]*num_seg_a + [1]*num_seg_b

            # There should be a segment_id for every input token.
            assert len(segment_ids) == len(input_ids)

            #Run our example through the model.
            start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                                  return_dict=False)

            # Find the tokens with the highest `start` and `end` scores.
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            answer = ' '.join(tokens[answer_start:answer_end+1])
            corrected_answer = ''

            for word in answer.split():
                if word[0:2] == '##':
                    corrected_answer += word[2:]
                else:
                    corrected_answer += ' ' + word

            await message.answer(corrected_answer)
        await asyncio.sleep(2)
        while True:
            if message.message_id > mid:
                mid = message.message_id
                textin = message.text
                break
            else:
                await asyncio.sleep(1) 
    dp.stop_polling()
    await dp.wait_closed()
    await bot.session.close()

if __name__ == '__main__':
    app.run()
    loop.create_task(executor.start_polling(dp, skip_updates=True))
    loop.run_until_complete(main())
    loop.stop()
    loop.close()


    


