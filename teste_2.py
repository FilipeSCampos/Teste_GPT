import torch
from Teste_GPT.gpt_teste import TransformerModel  # Importe o seu módulo Transformer personalizado
from transformers import GPT2Tokenizer
# Carregue o modelo treinado
model = TransformerModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Função de geração de texto
def gerar_texto(seed_text, max_length=50):
    # Crie uma instância do tokenizador
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize a semente de entrada
    input_tokens = tokenizer.encode(seed_text, add_special_tokens=False, return_tensors='pt')
    input_ids = input_tokens

    # Inicialize a saída com a semente
    generated_text = input_tokens

    with torch.no_grad():
        for _ in range(max_length):
            # Gere o próximo token
            output = model(input_ids, input_ids)
            predicted_token_id = torch.argmax(output[0, -1, :]).item()
            predicted_token = tokenizer.decode(predicted_token_id)

            # Adicione o token gerado à sequência de saída
            generated_text = torch.cat((generated_text, torch.tensor([[predicted_token_id]])), dim=1)

            # Pare se encontrar um token de fim de sequência
            if predicted_token == '[SEP]' or predicted_token == '[PAD]':
                break

    generated_text = tokenizer.decode(generated_text[0])
    return generated_text

# Gere texto a partir de uma semente
seed_text = "Uma vez, em uma terra distante,"
generated_text = gerar_texto(seed_text)
print(generated_text)
