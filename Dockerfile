FROM python:3.10

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=7860

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
